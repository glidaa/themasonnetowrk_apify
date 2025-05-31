import asyncio
import logging
import httpx
from bs4 import BeautifulSoup
from apify import Actor
import datetime
from urllib.parse import urljoin, urlparse
import os
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions for Stage 2 ---

async def check_iframe_compatibility(url: str) -> bool:
    """
    Checks if a URL is likely to render in an iframe based on HTTP headers.
    Returns True if likely to render, False if likely to be blocked.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.head(url, timeout=10)
            response.raise_for_status()

            x_frame_options = response.headers.get('X-Frame-Options', '').lower()
            if x_frame_options == 'deny' or x_frame_options == 'sameorigin':
                logging.info(f"  URL '{url}' blocked by X-Frame-Options: {x_frame_options}")
                return False

            csp = response.headers.get('Content-Security-Policy', '')
            if 'frame-ancestors' in csp:
                if 'frame-ancestors \'none\'' in csp or 'frame-ancestors \'src\'' in csp:
                     logging.info(f"  URL '{url}' likely blocked by CSP frame-ancestors.")
                     return False

    except httpx.HTTPStatusError as e:
        logging.warning(f"  HTTP error checking iframe compatibility for {url}: {e.response.status_code}")
        return False
    except httpx.RequestError as e:
        logging.warning(f"  Network error checking iframe compatibility for {url}: {e}")
        return False
    except Exception as e:
        logging.warning(f"  Unexpected error checking iframe compatibility for {url}: {e}")
        return False

    logging.info(f"  URL '{url}' seems iframe compatible (based on headers).")
    return True

async def get_page_main_content(url: str) -> str:
    """
    Fetches the content of a URL and attempts to extract the main article/body text.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Common selectors for main article content. You might need to refine this for specific sites.
            content_selectors = [
                'article', 'main', '.main-content', '#main-content', '.article-body',
                'div[role="main"]', 'div.story-content', 'div.entry-content',
                'div.post-content', 'div.content-body', 'body' # 'body' is a fallback, can be noisy
            ]

            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    # Remove script, style, navigation, footer, header, and sidebar tags to clean up text
                    for unwanted_tag in element(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        unwanted_tag.decompose()
                    return element.get_text(separator=' ', strip=True)

            return soup.body.get_text(separator=' ', strip=True) if soup.body else ''

    except httpx.HTTPStatusError as e:
        logging.error(f"  HTTP error fetching content for {url}: {e.response.status_code}")
        return ""
    except httpx.RequestError as e:
        logging.error(f"  Network error fetching content for {url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"  Unexpected error getting content for {url}: {e}")
        return ""

async def generate_story_with_openai(text_content: str, api_key: str) -> str:
    """
    Sends text content to OpenAI's API to generate a detailed story.
    """
    if not text_content:
        return "[No content to summarize/generate story from]"

    client = openai.AsyncOpenAI(api_key=api_key)

    # --- CHANGED: Prompt and input text limit for longer story ---
    prompt = (
        "Based on the following news article content, write a detailed and comprehensive "
        "news story (around 500-800 words) that captures all the main "
        "points, context, and implications. Focus on the core facts and provide a narrative. "
        "Do not include a title, just the story body.\n\n"
        f"Article Content:\n{text_content[:8000]}" # Increased input text limit to 8000 characters
    )

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful news reporter."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo", # Consider "gpt-4" for higher quality and better adherence to length, but at higher cost
            temperature=0.7,
            max_tokens=1000 # Increased max_tokens to allow for a longer generated story (approx 750 words)
        )
        return chat_completion.choices[0].message.content.strip()

    except openai.APIConnectionError as e:
        logging.error(f"  OpenAI API connection error: {e}")
        return "[OpenAI API connection error]"
    except openai.RateLimitError as e:
        logging.error(f"  OpenAI API rate limit exceeded: {e}")
        return "[OpenAI API rate limit exceeded]"
    except openai.APIStatusError as e:
        logging.error(f"  OpenAI API error (Status {e.status_code}): {e.response}")
        return "[OpenAI API error]"
    except Exception as e:
        logging.error(f"  General error with OpenAI API: {e}")
        return "[Error generating story with OpenAI]"


# --- Main Actor Logic ---
async def main():
    async with Actor:
        logging.info("Starting Drudge Report Scraper (Beautiful Soup - Dedicated)...")
        drudge_url = "https://www.drudgereport.com/"

        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logging.error("OPENAI_API_KEY environment variable is not set. Cannot use OpenAI API.")

        # --- Stage 1: Scrape Drudge Report and save initial links ---
        initial_drudge_links = []
        try:
            async with httpx.AsyncClient() as client:
                logging.info(f"Fetching Drudge Report from: {drudge_url}")
                response = await client.get(drudge_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                logging.info("Drudge Report page fetched successfully.")

            headline_selectors = ['a > b', 'font[size="+2"] a', 'font[size="+1"] a', 'a b']
            extracted_headline_texts = set()

            for selector in headline_selectors:
                for element in soup.select(selector):
                    text = element.get_text(strip=True)
                    href = element.get('href')
                    if text and text not in extracted_headline_texts:
                        await Actor.push_data(data={
                            "type": "headline",
                            "text": text,
                            "href": href,
                            "scrape_timestamp": datetime.datetime.now().isoformat()
                        })
                        extracted_headline_texts.add(text)
            logging.info(f"Extracted {len(extracted_headline_texts)} main headlines from Drudge.")
            if not extracted_headline_texts:
                logging.warning("WARNING: No main headlines extracted from Drudge. Check selectors.")

            excluded_text_keywords = [
                "link", "archives", "drudge", "search", "donate", "contact", "mail", "tip", "jobs", "advertise",
                "feedback", "sitemap", "privacy", "terms", "about"
            ]
            excluded_domains = [
                "drudgereport.com", "foxnews.com", "cnn.com", "reuters.com", "nytimes.com", "wsj.com",
                "breitbart.com", "dailymail.co.uk", "washingtontimes.com", "washingtonexaminer.com",
                "news.google.com", "apnews.com", "cbsnews.com", "abcnews.go.com", "nbcnews.com",
                "msnbc.com", "usatoday.com", "nypost.com", "thehill.com", "politico.com",
                "washingtonpost.com", "latimes.com", "sfgate.com", "chicagotribune.com",
                "weeklystandard.com", "nationalreview.com", "dailycaller.com",
                "realclearpolitics.com", "thegatewaypundit.com", "infowars.com", "rt.com",
                "sputniknews.com", "theguardian.com", "independent.co.uk", "bbc.co.uk",
                "apify.com", "googleusercontent.com", "twitter.com", "facebook.com",
                "instagram.com", "reddit.com", "telegram.org", "bitchute.com", "rumble.com",
                "trends.google.com", "pressreader.com", "news.sky.com", "boxofficemojo.com",
                "ustvdb.com", "abcnews.com", "theatlantic.com", "axios.com",
                "billboard.com", "boston.com", "bostonherald.com", "businessinsider.com",
                "cbslocal.com", "c-span.org", "suntimes.com", "csmonitor.com", "cnbc.com",
                "crazydaysandnights.net", "thedailybeast.com", "deadline.com", "elnuevodia.com",
                "ew.com", "thefp.com", "hollywoodreporter.com", "huffingtonpost.com",
                "theintercept.com", "dailynewslosangeles.com", "marketwatch.com",
                "mediaite.com", "motherjones.com", "thenation.com", "thenewrepublic.com",
                "nymag.com", "nydailynews.com", "newyorker.com", "newsmax.com",
                "newzit.com", "people.com", "rawstory.com", "reason.org",
                "rollcall.com", "rollingstone.com", "salon.com", "semafor.com",
                "showbiz411.com", "sky.com", "thesmokinggun.com", "tmz.com",
                "usnews.com", "vanityfair.com", "variety.com", "online.wsj.com",
                "worldofreel.com", "thewrap.com", "africanews.com", "arabnews.com",
                "bild.de", "en.people.cn", "clarin.com", "welt.de",
                "zeit.de", "economist.com", "elmundo.es", "english.elpais.com",
                "ft.com", "folha.uol.com.br", "wyborcza.pl", "gbnews.com",
                "haaretz.com", "hindustantimes.com", "hurriyetdailynews.com",
                "japantimes.co.jp", "jpost.com", "la-prensa.com.mx", "repubblica.it",
                "lefigaro.fr", "lemonde.fr", "lesoir.be", "themoscowtimes.com",
                "theneweuropean.co.uk", "asia.nikkei.com", "riotimesonline.com",
                "smh.com.au", "timesofindia.indiatimes.com", "mirror.co.uk",
                "dailystar.co.uk", "express.co.uk", "metro.co.uk", "standard.co.uk",
                "thesun.co.uk", "telegraph.co.uk", "thetimes.com", "japannews.yomiuri.co.jp",
                "france24.com", "player.streamguys.com",
                "bloomberg.com", "nordot.app", "dw.com", "interfax.com",
                "itar-tass.com", "english.kyodonews.net", "mcclatchydc.com",
                "nhk.or.jp", "pravdareport.com", "ptinews.com", "xinhuanet.com",
                "english.yonhapnews.co.kr", "drudgereportarchives.com", "zoom.earth",
                "theyeshivaworld.com"
            ]

            extracted_link_count = 0
            filtered_out_count = 0
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)

                absolute_href = href
                if href.startswith('http://') or href.startswith('https://') or href.startswith('//'):
                    pass
                elif href.startswith('#'):
                    continue
                else:
                    absolute_href = urljoin(drudge_url, href)

                should_exclude = False
                lower_text = text.lower()
                for keyword in excluded_text_keywords:
                    if keyword.lower() in lower_text:
                        should_exclude = True
                        break
                if should_exclude:
                    filtered_out_count += 1
                    continue

                try:
                    parsed_url = urlparse(absolute_href)
                    hostname = parsed_url.hostname.lower() if parsed_url.hostname else ''
                    if hostname.startswith('www.'): hostname = hostname[4:]
                    if hostname in excluded_domains:
                        should_exclude = True
                except ValueError:
                    logging.warning(f"Could not parse URL for exclusion check: {absolute_href}")
                    pass
                if should_exclude:
                    filtered_out_count += 1
                    continue

                initial_drudge_links.append({"text": text, "href": absolute_href})
                await Actor.push_data(data={
                    "type": "link",
                    "text": text if text else f"[Link to {absolute_href}]",
                    "href": absolute_href,
                    "scrape_timestamp": datetime.datetime.now().isoformat()
                })
                extracted_link_count += 1

            logging.info(f"Extracted {extracted_link_count} initial Drudge links after filtering.")
            logging.info(f"Filtered out {filtered_out_count} initial Drudge links.")

            if not openai_api_key:
                logging.warning("Skipping Stage 2 (OpenAI processing) because OPENAI_API_KEY is not set.")
            else:
                logging.info(f"Starting Stage 2: Processing {len(initial_drudge_links)} collected links...")
                processed_link_count = 0
                for link_info in initial_drudge_links:
                    url = link_info["href"]
                    original_text = link_info["text"]

                    logging.info(f"Checking link: {url}")
                    is_iframe_compatible = await check_iframe_compatibility(url)

                    if not is_iframe_compatible:
                        logging.info(f"  Link '{url}' is NOT iframe compatible. Attempting to generate story.")
                        article_content = await get_page_main_content(url)

                        if article_content:
                            generated_story = await generate_story_with_openai(article_content, openai_api_key)
                            await Actor.push_data(data={
                                "type": "generated_story",
                                "original_link_text": original_text,
                                "original_link_href": url,
                                "generated_story": generated_story,
                                "story_generation_timestamp": datetime.datetime.now().isoformat()
                            })
                            logging.info(f"  Generated and saved story for '{url}'.")
                        else:
                            logging.warning(f"  No content extracted from '{url}' for story generation.")
                    else:
                        logging.info(f"  Link '{url}' IS iframe compatible. Skipping story generation.")

                    processed_link_count += 1

                logging.info(f"Finished Stage 2: Processed {processed_link_count} links.")

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error during main scrape of Drudge Report: {e.response.status_code} {e.response.reason_phrase}")
        except httpx.RequestError as e:
            logging.error(f"Network error during main scrape of Drudge Report: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during main scrape: {e}")

if __name__ == '__main__':
    asyncio.run(main())