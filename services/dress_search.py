"""
Dress search service that searches the web for dress images based on user prompts
without using paid APIs - completely free and open source solution.
"""
import requests
from bs4 import BeautifulSoup
import re
import random
import time
import logging
import json
from urllib.parse import quote_plus, urlparse, urlencode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_image_url(url):
    """Check if a URL appears to be a valid image URL"""
    # Check if the URL has an image extension or contains image patterns
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Accept more image formats and URLs that might not have explicit extensions
    if not (path.endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')) or 
            'images' in url.lower() or 
            'img' in url.lower() or
            'photos' in url.lower()):
        return False
    
    # Check for suspicious or placeholder patterns
    suspicious_patterns = [
        'placeholder', 'spacer', 'blank', 'transparent', 'avatar',
        'default', 'empty', 'null', 'void', 'no-image', 'noimage',
        'icon', 'logo', 'button'
    ]
    
    for pattern in suspicious_patterns:
        if pattern in url.lower():
            return False
    
    # Check if URL has a reasonable length
    if len(url) < 15 or len(url) > 1000:
        return False
        
    return True

def search_dresses(query):
    """
    Search for dress images based on the user query.
    
    Args:
        query (str): User search term for dress type
        
    Returns:
        dict: Results containing query and list of image URLs with sources
    """
    if not query or len(query.strip()) == 0:
        return {"error": "No search query provided", "results": []}
        
    # Clean up and enhance the search query
    search_term = query.strip()
    
    # Don't automatically append 'dress' for clothing searches that aren't dresses
    clothing_keywords = ['suit', 'jacket', 'shirt', 'pants', 'trousers', 'coat', 'tuxedo']
    if 'dress' not in search_term.lower() and not any(keyword in search_term.lower() for keyword in clothing_keywords):
        search_term += ' dress'
    
    # Add fashion keyword to improve results quality
    if 'fashion' not in search_term.lower():
        search_term += ' fashion'
        
    # URL encode the search term
    encoded_term = quote_plus(search_term)
    
    # Use different User-Agent headers to avoid being blocked
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 Edg/98.0.1108.62',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:98.0) Gecko/20100101 Firefox/98.0'
    ]
    
    # Track search results from multiple sources
    image_results = []
    max_results_per_source = 50  # Increased from 30 to 50
    
    # Search source 1: DuckDuckGo images
    try:
        logger.info(f"Searching DuckDuckGo for '{search_term}'")
        
        # Use a random user agent
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://duckduckgo.com/'
        }
        
        # First request to get the VQD parameter
        url = f"https://duckduckgo.com/?q={encoded_term}&iax=images&ia=images"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the VQD token
            vqd_match = re.search(r'vqd=\'([\d-]+)\'', response.text)
            if vqd_match:
                vqd = vqd_match.group(1)
                
                # Now make a request to the JSON API endpoint
                json_url = f"https://duckduckgo.com/i.js?q={encoded_term}&o=json&vqd={vqd}"
                headers['Referer'] = url  # Set the referer to the search page
                
                json_response = requests.get(json_url, headers=headers, timeout=15)
                if json_response.status_code == 200:
                    try:
                        data = json_response.json()
                        if 'results' in data:
                            for result in data['results']:
                                if 'image' in result and len(image_results) < max_results_per_source:
                                    image_url = result.get('image')
                                    if image_url and is_valid_image_url(image_url):
                                        image_results.append({
                                            "url": image_url,
                                            "source": "DuckDuckGo",
                                            "filename": f"dress_{len(image_results) + 1}.jpg"
                                        })
                    except json.JSONDecodeError:
                        logger.error("Failed to parse DuckDuckGo JSON response")
        
            # Additional extraction from the HTML for more images
            scripts = soup.find_all('script')
            img_tags = soup.find_all('img')
            
            # Extract from scripts
            for script in scripts:
                if script.string:
                    img_matches = re.findall(r'(https://[^"\'\s]+\.(?:jpg|jpeg|png|webp|gif))', str(script.string))
                    for img_url in img_matches:
                        if len(image_results) < max_results_per_source and is_valid_image_url(img_url):
                            # Extract filename from URL or generate one
                            path = urlparse(img_url).path
                            filename = path.split("/")[-1] if "/" in path else f"dress_{len(image_results) + 1}.jpg"
                            
                            image_results.append({
                                "url": img_url,
                                "source": "DuckDuckGo",
                                "filename": filename
                            })
            
            # Extract from img tags
            for img in img_tags:
                src = img.get('src', '')
                if src.startswith('http') and is_valid_image_url(src):
                    image_results.append({
                        "url": src,
                        "source": "DuckDuckGo",
                        "filename": f"img_ddg_{len(image_results) + 1}.jpg"
                    })
        
        logger.info(f"Found {len(image_results)} images from DuckDuckGo")
    except Exception as e:
        logger.error(f"Error searching DuckDuckGo: {e}")
    
    # Add delay to avoid being blocked
    time.sleep(0.5)
    
    # Search source 2: Bing Images (add back Bing search)
    try:
        logger.info(f"Searching Bing for '{search_term}'")
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.bing.com/'
        }
        
        # Use different URL formats to get more results
        bing_urls = [
            f"https://www.bing.com/images/search?q={encoded_term}&form=HDRSC2&first=1",
            f"https://www.bing.com/images/search?q={encoded_term}&qft=+filterui:photo-photo&form=IRFLTR"
        ]
        
        for url in bing_urls:
            if len(image_results) >= max_results_per_source * 2:  # If we already have lots of results, skip
                break
                
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                logger.info("Bing search successful")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract image URLs from Bing's JSON data
                img_containers = soup.select('.imgpt')
                for container in img_containers:
                    # Look for m attribute which contains image data
                    m_attrs = container.select('[m]')
                    for el in m_attrs:
                        try:
                            m_data = json.loads(el['m'])
                            if 'murl' in m_data:
                                img_url = m_data['murl']
                                if is_valid_image_url(img_url) and len(image_results) < max_results_per_source * 2:
                                    image_results.append({
                                        "url": img_url,
                                        "source": "Bing",
                                        "filename": f"bing_{len(image_results) + 1}.jpg"
                                    })
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                # Additional extraction methods for Bing
                # Try another approach for Bing - look for all image tags
                image_tags = soup.select('img.mimg')
                for img in image_tags:
                    src = img.get('src', '')
                    if src.startswith('http') and is_valid_image_url(src) and len(image_results) < max_results_per_source * 2:
                        image_results.append({
                            "url": src,
                            "source": "Bing",
                            "filename": f"bing_img_{len(image_results) + 1}.jpg"
                        })
                        
                # Try to extract from JSON data in script tags
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string:
                        img_urls = re.findall(r'(https://[^"\']+\.(?:jpg|jpeg|png|webp|gif))', str(script.string))
                        for img_url in img_urls:
                            if is_valid_image_url(img_url) and len(image_results) < max_results_per_source * 2:
                                image_results.append({
                                    "url": img_url,
                                    "source": "Bing",
                                    "filename": f"bing_script_{len(image_results) + 1}.jpg"
                                })
                
        logger.info(f"Found {len(image_results)} images total after Bing search")
    except Exception as e:
        logger.error(f"Error searching Bing: {e}")
    
    # Add delay between searches
    time.sleep(0.5)
    
    # Search source 3: Unsplash API
    try:
        logger.info(f"Searching Unsplash for '{search_term}'")
        
        # Different path for Unsplash - we'll use their public search page
        unsplash_term = quote_plus(search_term)
        url = f"https://unsplash.com/s/photos/{unsplash_term}"
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://unsplash.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            logger.info("Unsplash search successful")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all image elements
            img_tags = soup.select('img[srcset], img[src]')
            for img in img_tags:
                if len(image_results) >= max_results_per_source:
                    break
                    
                # Try to extract from srcset attribute
                srcset = img.get('srcset', '')
                if srcset:
                    urls = re.findall(r'(https://[^\s]+)', srcset)
                    if urls:
                        img_url = urls[0].split(' ')[0]  # Get the URL part only
                        if is_valid_image_url(img_url):
                            image_results.append({
                                "url": img_url,
                                "source": "Unsplash",
                                "filename": f"unsplash_{len(image_results) + 1}.jpg"
                            })
                            continue
                
                # Fall back to src attribute if no srcset or srcset didn't yield valid results
                src = img.get('src', '')
                if src and src.startswith('http') and is_valid_image_url(src):
                    image_results.append({
                        "url": src,
                        "source": "Unsplash",
                        "filename": f"unsplash_{len(image_results) + 1}.jpg"
                    })
                    
            logger.info(f"Found {len(image_results)} images total after Unsplash search")
    except Exception as e:
        logger.error(f"Error searching Unsplash: {e}")
    
    # Add delay between searches
    time.sleep(0.5)
    
    # Search source 4: Google Search images
    try:
        logger.info(f"Searching Google for '{search_term}'")
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }
        
        url = f"https://www.google.com/search?q={encoded_term}&tbm=isch"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            logger.info("Google search successful")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Google stores image data in special script tags
            script_data = re.findall(r'AF_initDataCallback\((.*?)\);', response.text, re.DOTALL)
            for script in script_data:
                # Extract image URLs from the data
                img_urls = re.findall(r'(https://[^"\']+\.(?:jpg|jpeg|png|webp|gif))', script)
                for img_url in img_urls:
                    if is_valid_image_url(img_url) and len(image_results) < max_results_per_source:
                        # Generate a filename
                        filename = f"google_{len(image_results) + 1}.jpg"
                        
                        image_results.append({
                            "url": img_url,
                            "source": "Google",
                            "filename": filename
                        })
            
            # Also try to extract from img tags
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if src.startswith('http') and is_valid_image_url(src) and len(image_results) < max_results_per_source:
                    image_results.append({
                        "url": src,
                        "source": "Google",
                        "filename": f"google_img_{len(image_results) + 1}.jpg"
                    })
            
            logger.info(f"Found {len(image_results)} images total after Google search")
    except Exception as e:
        logger.error(f"Error searching Google: {e}")
    
    # Add delay between searches
    time.sleep(0.5)
    
    # Search source 5: Pinterest (often has lots of fashion images)
    try:
        logger.info(f"Searching Pinterest for '{search_term}'")
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.pinterest.com/'
        }
        
        url = f"https://www.pinterest.com/search/pins/?q={encoded_term}"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            logger.info("Pinterest search successful")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract from img tags
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if src and src.startswith('http') and is_valid_image_url(src) and len(image_results) < max_results_per_source:
                    image_results.append({
                        "url": src,
                        "source": "Pinterest",
                        "filename": f"pinterest_{len(image_results) + 1}.jpg"
                    })
            
            # Pinterest stores data in JSON in script tags
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'initial_state' in str(script.string):
                    # Extract image URLs
                    img_urls = re.findall(r'(https://[^"\']+\.(?:jpg|jpeg|png|webp|gif))', str(script.string))
                    for img_url in img_urls:
                        if is_valid_image_url(img_url) and len(image_results) < max_results_per_source:
                            image_results.append({
                                "url": img_url,
                                "source": "Pinterest",
                                "filename": f"pinterest_pin_{len(image_results) + 1}.jpg"
                            })
            
            logger.info(f"Found {len(image_results)} images total after Pinterest search")
    except Exception as e:
        logger.error(f"Error searching Pinterest: {e}")
    
    # Add source 6: Yahoo Image Search (add another source)
    try:
        logger.info(f"Searching Yahoo for '{search_term}'")
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://search.yahoo.com/'
        }
        
        url = f"https://images.search.yahoo.com/search/images?p={encoded_term}"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            logger.info("Yahoo search successful")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs from the page
            images = soup.select('img')
            for img in images:
                src = img.get('src', '')
                if src and src.startswith('http') and is_valid_image_url(src):
                    image_results.append({
                        "url": src,
                        "source": "Yahoo",
                        "filename": f"yahoo_{len(image_results) + 1}.jpg"
                    })
            
            # Extract from data attributes
            images_with_data = soup.select('[data-src]')
            for img in images_with_data:
                src = img.get('data-src', '')
                if src and src.startswith('http') and is_valid_image_url(src):
                    image_results.append({
                        "url": src,
                        "source": "Yahoo",
                        "filename": f"yahoo_data_{len(image_results) + 1}.jpg"
                    })
            
            logger.info(f"Found {len(image_results)} images total after Yahoo search")
    except Exception as e:
        logger.error(f"Error searching Yahoo: {e}")
    
    # Add fallback stock images if we have very few results
    if len(image_results) < 5:
        logger.info("Adding fallback images due to low result count")
        
        # Dictionary mapping clothing types to fallback images
        fallbacks = {
            'suit': [
                'https://images.unsplash.com/photo-1553240779-e6b4d57c4224',
                'https://images.unsplash.com/photo-1593032465175-481ac7f401f0',
                'https://images.unsplash.com/photo-1507679799987-c73779587ccf',
                'https://images.unsplash.com/photo-1617127365659-c47fa864d8bc',
                'https://images.unsplash.com/photo-1598808503479-76b74e2261a7'
            ],
            'dress': [
                'https://images.unsplash.com/photo-1595777457583-95e059d581b8',
                'https://images.unsplash.com/photo-1566174053879-31528523f8ae',
                'https://images.unsplash.com/photo-1597923206041-08a4ea688848',
                'https://images.unsplash.com/photo-1502727135886-df285cc8379f',
                'https://images.unsplash.com/photo-1613915617430-8ab0fd7c6baf'
            ],
            'shirt': [
                'https://images.unsplash.com/photo-1603252109303-2751441dd157',
                'https://images.unsplash.com/photo-1588359348347-9bc6cbbb689e',
                'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf',
                'https://images.unsplash.com/photo-1596755094514-f87e34085b2c'
            ],
            'jacket': [
                'https://images.unsplash.com/photo-1591047139829-d91aecb6caea',
                'https://images.unsplash.com/photo-1548883354-94bcfe321cbb',
                'https://images.unsplash.com/photo-1551028719-00167b16eac5',
                'https://images.unsplash.com/photo-1559551409-dadc959f76b8'
            ]
        }
        
        # Look for matches in the query
        for key, urls in fallbacks.items():
            if key in search_term.lower():
                for i, url in enumerate(urls):
                    image_results.append({
                        'url': url,
                        'source': 'Curated Stock Images',
                        'filename': f"fallback_{key}_{i+1}.jpg"
                    })
        
        # Add some general clothing images if no specific match found
        if len(image_results) < 5:
            general_clothes = [
                'https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5',
                'https://images.unsplash.com/photo-1516762689617-e1cffcef479d',
                'https://images.unsplash.com/photo-1441986300917-64674bd600d8',
                'https://images.unsplash.com/photo-1479064555552-3ef4979f8908',
                'https://images.unsplash.com/photo-1558769132-cb1aea458c5e',
                'https://images.unsplash.com/photo-1539008835657-9e8e9680c956',
                'https://images.unsplash.com/photo-1551232864-3f0890e580d9',
                'https://images.unsplash.com/photo-1542272604-787c3835535d'
            ]
            
            for i, url in enumerate(general_clothes):
                image_results.append({
                    'url': url,
                    'source': 'General Fashion Images',
                    'filename': f"general_fashion_{i+1}.jpg"
                })
    
    # De-duplicate results but preserve as many unique images as possible
    seen_urls = set()
    unique_results = []
    
    for result in image_results:
        url = result['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    # Ensure we have sufficient results
    if len(unique_results) < 5:
        logger.warning(f"Very few results found ({len(unique_results)}) for query: '{query}'")
    
    # Shuffle results to mix sources
    random.shuffle(unique_results)
    
    # Limit results to a reasonable number but ensure we have enough
    max_display_results = 30
    final_results = unique_results[:max_display_results]
    
    # Log search completion
    logger.info(f"Search complete for '{query}', found {len(final_results)} results")
    
    return {
        "query": query,
        "results": final_results
    }
