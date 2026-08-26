import re
import urllib
import json
import requests
import tldextract

from urllib.parse import urlparse, urlencode
from bs4 import BeautifulSoup

key = 'g08gow00ok4c4o0wocko8kkkok040okcsg0k0oso'

HINTS = ['wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view']


def url_length(url):
    return len(url)

def get_domain(url):
    o = urllib.parse.urlsplit(url)
    return o.hostname, tldextract.extract(url).domain, o.path

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '[0-9a-fA-F]{7}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
    
def count_dots(hostname):
    return hostname.count('.')

def count_exclamination(base_url):
    return base_url.count('?')

def count_equal(base_url):
    return base_url.count('=')

def count_slash(full_url):
    return full_url.count('/')

def check_www(words_raw):
    count = 0
    for word in words_raw:
        if not word.find('www') == -1:
            count += 1
    return count

def ratio_digits(url):
    return len(re.sub("[^0-9]", "", url))/len(url)

def tld_in_subdomain(tld, subdomain):
    if subdomain.count(tld)>0:
        return 1
    return 0

def prefix_suffix(hostname):
    if re.findall(r"https?://[^\-]+-[^\-]+/", hostname):
        return 1
    else:
        return 0 

def shortest_word_length(words_raw):
    if len(words_raw) ==0:
        return 0
    return min(len(word) for word in words_raw)

def longest_word_length(words_raw):
    if len(words_raw) ==0:
        return 0
    return max(len(word) for word in words_raw) 

def phish_hints(url_path):
    count = 0
    for hint in HINTS:
        count += url_path.lower().count(hint)
    return count

def nb_hyperlinks(dom):
    return len(dom.find("href")) + len(dom.find("src"))

def h_total(Href, Link, Media, Form, CSS, Favicon):
    return len(Href['internals']) + len(Href['externals']) + len(Link['internals']) + len(Link['externals']) + \
           len(Media['internals']) + len(Media['externals']) + len(Form['internals']) + len(Form['externals']) + \
           len(CSS['internals']) + len(CSS['externals']) + len(Favicon['internals']) + len(Favicon['externals'])

def h_internal(Href, Link, Media, Form, CSS, Favicon):
    return len(Href['internals']) + len(Link['internals']) + len(Media['internals']) +\
           len(Form['internals']) + len(CSS['internals']) + len(Favicon['internals'])


def internal_hyperlinks(Href, Link, Media, Form, CSS, Favicon):
    total = h_total(Href, Link, Media, Form, CSS, Favicon)
    if total == 0:
        return 0
    else :
        return h_internal(Href, Link, Media, Form, CSS, Favicon)/total
    
def empty_title(Title):
    if Title:
        return 0
    return 1

def domain_in_title(domain, title):
    if domain.lower() in title.lower():
        return 0
    return 1

def domain_age(domain):
    url = domain.split("//")[-1].split("/")[0].split('?')[0]
    show = "https://input.payapi.io/v1/api/fraud/domain/age/" + url
    try:
        r = requests.get(show, timeout=10, verify=False)  # Added timeout and disabled SSL verification
        if r.status_code == 200:
            data = r.text
            jsonToPython = json.loads(data)
            result = jsonToPython['result']
            if result == None:
                return -2
            else:
                return result
        else:       
            return -1
    except Exception as e:
        print(f"Error getting domain age: {e}")
        return -1
    
def google_index(url):
    #time.sleep(.6)
    user_agent =  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
    headers = {'User-Agent' : user_agent}
    query = {'q': 'site:' + url}
    google = "https://www.google.com/search?" + urlencode(query)
    data = requests.get(google, headers=headers)
    data.encoding = 'ISO-8859-1'
    soup = BeautifulSoup(str(data.content), "html.parser")
    try:
        if 'Our systems have detected unusual traffic from your computer network.' in str(soup):
            return -1
        check = soup.find(id="rso").find("div").find("div").find("a")
        #print(check)
        if check and check['href']:
            return 0
        else:
            return 1
        
    except AttributeError:
        return 1

def page_rank(key, domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = requests.get(url, headers={'API-OPR':key})
        result = request.json()
        result = result['response'][0]['page_rank_integer']
        if result:
            return result
        else:
            return 0
    except:
        return -1



def words_raw_extraction(domain, subdomain, path):
    w_domain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
    w_subdomain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())   
    w_path = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
    raw_words = w_domain + w_path + w_subdomain
    w_host = w_domain + w_subdomain
    raw_words = list(filter(None,raw_words))
    return raw_words, list(filter(None,w_host)), list(filter(None,w_path))

def is_URL_accessible(url):
    """Check if URL is accessible and return page content"""
    page = None
    original_url = url
    
    # Common headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # Special headers for Indonesian websites
    id_headers = headers.copy()
    id_headers['Accept-Language'] = 'id-ID,id;q=0.9,en;q=0.8'
    
    # Try different URL variations
    url_variations = []
    
    # Add original URL
    url_variations.append(url)
    
    # Add www version if not present
    parsed = urlparse(url)
    if parsed.netloc and not parsed.netloc.startswith('www.'):
        url_variations.append(parsed._replace(netloc='www.' + parsed.netloc).geturl())
    
    # Add HTTPS version if HTTP
    if url.startswith('http://'):
        url_variations.append(url.replace('http://', 'https://'))
    
    # Add HTTP version if HTTPS
    if url.startswith('https://'):
        url_variations.append(url.replace('https://', 'http://'))
    
    # Try each variation with different approaches
    for test_url in url_variations:
        try:
            # First try with standard headers
            page = requests.get(test_url, timeout=15, verify=False, allow_redirects=True, headers=headers)
            
            # Accept more status codes (200, 201, 202, 203, 206)
            if page.status_code in [200, 201, 202, 203, 206] and page.content and len(page.content) > 0:
                return True, test_url, page
            
            # If we get a redirect (3xx), follow it
            elif page.status_code in [301, 302, 303, 307, 308]:
                if 'location' in page.headers:
                    redirect_url = page.headers['location']
                    try:
                        redirect_page = requests.get(redirect_url, timeout=15, verify=False, allow_redirects=True, headers=headers)
                        if redirect_page.status_code in [200, 201, 202, 203, 206] and redirect_page.content and len(redirect_page.content) > 0:
                            return True, redirect_url, redirect_page
                    except:
                        pass
            
        except requests.exceptions.SSLError:
            # Try without SSL verification
            try:
                page = requests.get(test_url, timeout=15, verify=False, allow_redirects=True, headers=headers)
                if page.status_code in [200, 201, 202, 203, 206] and page.content and len(page.content) > 0:
                    return True, test_url, page
            except:
                pass
        except Exception as e:
            print(f"Failed to access {test_url}: {e}")
            continue
    
    # Try with Indonesian headers for .id domains
    if '.id' in url:
        for test_url in url_variations:
            try:
                page = requests.get(test_url, timeout=15, verify=False, allow_redirects=True, headers=id_headers)
                if page.status_code in [200, 201, 202, 203, 206] and page.content and len(page.content) > 0:
                    return True, test_url, page
            except Exception as e:
                print(f"Failed to access {test_url} with ID headers: {e}")
                continue
    
    # If all variations fail, try one more time with session (handles cookies better)
    try:
        session = requests.Session()
        session.headers.update(headers)
        page = session.get(url, timeout=15, verify=False, allow_redirects=True)
        if page.status_code in [200, 201, 202, 203, 206] and page.content and len(page.content) > 0:
            return True, url, page
    except Exception as e:
        print(f"Session request failed for {url}: {e}")
    
    return False, None, None
        
def extract_data_from_URL(hostname, content, domain, Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
               "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    soup = BeautifulSoup(content, 'html.parser', from_encoding='iso-8859-1')

    # collect all external and internal hrefs from url
    for href in soup.find_all('a', href=True):
        dots = [x.start(0) for x in re.finditer('\.', href['href'])]
        if hostname in href['href'] or domain in href['href'] or len(dots) == 1 or not href['href'].startswith('http'):
            if "#" in href['href'] or "javascript" in href['href'].lower() or "mailto" in href['href'].lower():
                 Anchor['unsafe'].append(href['href']) 
            if not href['href'].startswith('http'):
                if not href['href'].startswith('/'):
                    Href['internals'].append(hostname+'/'+href['href']) 
                elif href['href'] in Null_format:
                    Href['null'].append(href['href'])  
                else:
                    Href['internals'].append(hostname+href['href'])   
        else:
            Href['externals'].append(href['href'])
            Anchor['safe'].append(href['href'])

    # collect all media src tags
    for img in soup.find_all('img', src=True):
        dots = [x.start(0) for x in re.finditer('\.', img['src'])]
        if hostname in img['src'] or domain in img['src'] or len(dots) == 1 or not img['src'].startswith('http'):
            if not img['src'].startswith('http'):
                if not img['src'].startswith('/'):
                    Media['internals'].append(hostname+'/'+img['src']) 
                elif img['src'] in Null_format:
                    Media['null'].append(img['src'])  
                else:
                    Media['internals'].append(hostname+img['src'])   
        else:
            Media['externals'].append(img['src'])
           
    
    for audio in soup.find_all('audio', src=True):
        dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
        if hostname in audio['src'] or domain in audio['src'] or len(dots) == 1 or not audio['src'].startswith('http'):
             if not audio['src'].startswith('http'):
                if not audio['src'].startswith('/'):
                    Media['internals'].append(hostname+'/'+audio['src']) 
                elif audio['src'] in Null_format:
                    Media['null'].append(audio['src'])  
                else:
                    Media['internals'].append(hostname+audio['src'])   
        else:
            Media['externals'].append(audio['src'])
            
    for embed in soup.find_all('embed', src=True):
        dots = [x.start(0) for x in re.finditer('\.', embed['src'])]
        if hostname in embed['src'] or domain in embed['src'] or len(dots) == 1 or not embed['src'].startswith('http'):
             if not embed['src'].startswith('http'):
                if not embed['src'].startswith('/'):
                    Media['internals'].append(hostname+'/'+embed['src']) 
                elif embed['src'] in Null_format:
                    Media['null'].append(embed['src'])  
                else:
                    Media['internals'].append(hostname+embed['src'])   
        else:
            Media['externals'].append(embed['src'])
           
    for i_frame in soup.find_all('iframe', src=True):
        dots = [x.start(0) for x in re.finditer('\.', i_frame['src'])]
        if hostname in i_frame['src'] or domain in i_frame['src'] or len(dots) == 1 or not i_frame['src'].startswith('http'):
            if not i_frame['src'].startswith('http'):
                if not i_frame['src'].startswith('/'):
                    Media['internals'].append(hostname+'/'+i_frame['src']) 
                elif i_frame['src'] in Null_format:
                    Media['null'].append(i_frame['src'])  
                else:
                    Media['internals'].append(hostname+i_frame['src'])   
        else: 
            Media['externals'].append(i_frame['src'])
           

    # collect all link tags
    for link in soup.find_all('link', href=True):
        dots = [x.start(0) for x in re.finditer('\.', link['href'])]
        if hostname in link['href'] or domain in link['href'] or len(dots) == 1 or not link['href'].startswith('http'):
            if not link['href'].startswith('http'):
                if not link['href'].startswith('/'):
                    Link['internals'].append(hostname+'/'+link['href']) 
                elif link['href'] in Null_format:
                    Link['null'].append(link['href'])  
                else:
                    Link['internals'].append(hostname+link['href'])   
        else:
            Link['externals'].append(link['href'])

    for script in soup.find_all('script', src=True):
        dots = [x.start(0) for x in re.finditer('\.', script['src'])]
        if hostname in script['src'] or domain in script['src'] or len(dots) == 1 or not script['src'].startswith('http'):
            if not script['src'].startswith('http'):
                if not script['src'].startswith('/'):
                    Link['internals'].append(hostname+'/'+script['src']) 
                elif script['src'] in Null_format:
                    Link['null'].append(script['src'])  
                else:
                    Link['internals'].append(hostname+script['src'])   
        else:
            Link['externals'].append(link['href'])
           
            
    # collect all css
    for link in soup.find_all('link', rel='stylesheet'):
        dots = [x.start(0) for x in re.finditer('\.', link['href'])]
        if hostname in link['href'] or domain in link['href'] or len(dots) == 1 or not link['href'].startswith('http'):
            if not link['href'].startswith('http'):
                if not link['href'].startswith('/'):
                    CSS['internals'].append(hostname+'/'+link['href']) 
                elif link['href'] in Null_format:
                    CSS['null'].append(link['href'])  
                else:
                    CSS['internals'].append(hostname+link['href'])   
        else:
            CSS['externals'].append(link['href'])
    
    for style in soup.find_all('style', type='text/css'):
        try: 
            start = str(style[0]).index('@import url(')
            end = str(style[0]).index(')')
            css = str(style[0])[start+12:end]
            dots = [x.start(0) for x in re.finditer('\.', css)]
            if hostname in css or domain in css or len(dots) == 1 or not css.startswith('http'):
                if not css.startswith('http'):
                    if not css.startswith('/'):
                        CSS['internals'].append(hostname+'/'+css) 
                    elif css in Null_format:
                        CSS['null'].append(css)  
                    else:
                        CSS['internals'].append(hostname+css)   
            else: 
                CSS['externals'].append(css)
        except:
            continue
            
    # collect all form actions
    for form in soup.find_all('form', action=True):
        dots = [x.start(0) for x in re.finditer('\.', form['action'])]
        if hostname in form['action'] or domain in form['action'] or len(dots) == 1 or not form['action'].startswith('http'):
            if not form['action'].startswith('http'):
                if not form['action'].startswith('/'):
                    Form['internals'].append(hostname+'/'+form['action']) 
                elif form['action'] in Null_format or form['action'] == 'about:blank':
                    Form['null'].append(form['action'])  
                else:
                    Form['internals'].append(hostname+form['action'])   
        else:
            Form['externals'].append(form['action'])
            

    # collect all link tags
    for head in soup.find_all('head'):
        for head.link in soup.find_all('link', href=True):
            dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
            if hostname in head.link['href'] or len(dots) == 1 or domain in head.link['href'] or not head.link['href'].startswith('http'):
                if not head.link['href'].startswith('http'):
                    if not head.link['href'].startswith('/'):
                        Favicon['internals'].append(hostname+'/'+head.link['href']) 
                    elif head.link['href'] in Null_format:
                        Favicon['null'].append(head.link['href'])  
                    else:
                        Favicon['internals'].append(hostname+head.link['href'])   
            else:
                Favicon['externals'].append(head.link['href'])
                
        for head.link in soup.find_all('link', {'href': True, 'rel':True}):
            isicon = False
            if isinstance(head.link['rel'], list):
                for e_rel in head.link['rel']:
                    if (e_rel.endswith('icon')):
                        isicon = True
            else:
                if (head.link['rel'].endswith('icon')):
                    isicon = True
       
            if isicon:
                 dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                 if hostname in head.link['href'] or len(dots) == 1 or domain in head.link['href'] or not head.link['href'].startswith('http'):
                     if not head.link['href'].startswith('http'):
                        if not head.link['href'].startswith('/'):
                            Favicon['internals'].append(hostname+'/'+head.link['href']) 
                        elif head.link['href'] in Null_format:
                            Favicon['null'].append(head.link['href'])  
                        else:
                            Favicon['internals'].append(hostname+head.link['href'])   
                 else:
                     Favicon['externals'].append(head.link['href'])
                     
                    
    # collect i_frame
    for i_frame in soup.find_all('iframe', width=True, height=True, frameborder=True):
        if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['frameborder'] == "0":
            IFrame['invisible'].append(i_frame)
        else:
            IFrame['visible'].append(i_frame)
    for i_frame in soup.find_all('iframe', width=True, height=True, border=True):
        if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['border'] == "0":
            IFrame['invisible'].append(i_frame)
        else:
            IFrame['visible'].append(i_frame)
    for i_frame in soup.find_all('iframe', width=True, height=True, style=True):
        if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['style'] == "border:none;":
            IFrame['invisible'].append(i_frame)
        else:
            IFrame['visible'].append(i_frame)
          
    # get page title
    try:
        Title = soup.title.string
    except:
        pass
    
    # get content text
    Text = soup.get_text()

    # extract body text from <p> tags, limited to 1500 characters
    paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
    Body = ' '.join(paragraphs)[:1500]
    
    return Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text, Body


# url = 'https://supports1.umpikuja.com/'

# Href = {'internals':[], 'externals':[], 'null':[]}
# Link = {'internals':[], 'externals':[], 'null':[]}
# Anchor = {'safe':[], 'unsafe':[], 'null':[]}
# Media = {'internals':[], 'externals':[], 'null':[]}
# Form = {'internals':[], 'externals':[], 'null':[]}
# CSS = {'internals':[], 'externals':[], 'null':[]}
# Favicon = {'internals':[], 'externals':[], 'null':[]}
# IFrame = {'visible':[], 'invisible':[], 'null':[]}
# Title =''
# Text= ''

# state, url_accessible, page = is_URL_accessible(url)

# print('URL accessible: ', state)
# print('URL accessible URL: ', url_accessible)

# hostname, domain, path = get_domain(url)
# extracted_domain = tldextract.extract(url)
# domain = extracted_domain.domain+'.'+extracted_domain.suffix
# subdomain = extracted_domain.subdomain
# tmp = url[url.find(extracted_domain.suffix):len(url)]
# pth = tmp.partition("/")
# path = pth[1] + pth[2]
# words_raw, words_raw_host, words_raw_path= words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
# tld = extracted_domain.suffix
# parsed = urlparse(url)
# scheme = parsed.scheme

# if state:
#     content = page.content

#     Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text = extract_data_from_URL(hostname, content, domain, Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text)

# len_url = url_length(url)
# print('len_url: ',len_url)

# len_hostname = get_domain(url)[0]
# print('len_hostname: ', len(len_hostname))

# ip_address = having_ip_address(url)
# print('ip_address: ', ip_address)

# count_dots = count_dots(url)
# print('dots: ', count_dots)

# count_exclamination = count_exclamination(url)
# print('exclamation: ',count_exclamination)

# count_equal = count_equal(url)
# print('equal: ', count_equal)

# count_slash = count_slash(url)
# print('slash: ', count_slash)

# check_www = check_www(words_raw)
# print('www: ', check_www)

# ratio_digits_url = ratio_digits(url)
# print('ratio: ', ratio_digits_url)

# ratio_digits_host = ratio_digits(hostname)
# print('ratio_digits_host: ', ratio_digits_host)

# tld_in_subdomain = tld_in_subdomain(tld, subdomain)
# print('tld_in_subdomain: ', tld_in_subdomain)

# prefix_suffix = prefix_suffix(hostname)
# print('prefix_suffix: ', prefix_suffix)

# shortest = shortest_word_length(words_raw)
# print('shortest:' , shortest)

# longest_word = longest_word_length(words_raw)
# print('longest: ', longest_word)

# longest_path = longest_word_length(words_raw_path)
# print('longest_path: ', longest_path)

# phish_hints = phish_hints(url)
# print('phish_hints: ', phish_hints)

# nb_hyperlinks = h_total(Href, Link, Media, Form, CSS, Favicon)
# print('nb_hyperlinks: ', nb_hyperlinks)

# int_hyperlinks = internal_hyperlinks(Href, Link, Media, Form, CSS, Favicon)
# print('int_hyperlinks: ', int_hyperlinks)

# empty_title = empty_title(Title)
# print('empty_title: ', empty_title)

# domain_in_title = domain_in_title(domain, Title)
# print('domain_in_title: ', domain_in_title)

# domain_age = domain_age(domain)
# print('domain_age: ', domain_age)

# google_index = google_index(url)
# print('google_index: ', google_index)

# page_rank = page_rank(key, domain)
# print('page_rank: ', page_rank)
# Note: To use page_rank, you need an API key from https://www.domcop.com/openpagerank/
# Uncomment the lines above and define 'key' with your API key

# ... (existing code above)

# if __name__ == "__main__":
#     # Use the test URL already defined above, or set your own
#     url = 'https://digimo.id/fina-purwoko/?to=Fariska'

#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         html = response.text
#     except Exception as e:
#         print(f"Error fetching URL: {e}")
#         html = ""

#     if html:
#         soup = BeautifulSoup(html, 'html.parser')

#         forms = [str(form) for form in soup.find_all('form')]
#         heads = [str(head) for head in soup.find_all('head')]
#         titles = [title.get_text() for title in soup.find_all('title')]
#         scripts = [script.get_text() for script in soup.find_all('script')]

#         extracted_content = {
#             "forms": forms,
#             "heads": heads,
#             "titles": titles,
#             "scripts": scripts
#         }

#         with open("extracted_content4.json", "w", encoding="utf-8") as f:
#             json.dump(extracted_content, f, ensure_ascii=False, indent=2)
#         print("Extracted content saved to extracted_content.json")