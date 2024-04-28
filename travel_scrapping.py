import requests
from bs4 import BeautifulSoup
import json
import csv
import os


def scrape_articlesLinks(url):
    url="https://theworldtravelguy.com/blog/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('h2', class_='penci-entry-title entry-title grid-title')
    if articles is not None:
        links = [a.find('a', href=True)['href'] for a in articles]
        return links

    else:
        print("No articles found.")
# <ul class="penci-wrapper-data penci-grid penci-shortcode-render">
# <h2 class="penci-entry-title entry-title grid-title">
def check_links(links):
    processed_links = []
    for link in links:
        if link == "https:#" or link.startswith("https://theworldtravelguy"):
            continue
        processed_links.append(link)
    return processed_links


def article_data(url):
    article = {}
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # article_content = soup.find('div', class_='theiaStickySidebar')
        # if article_content:
        article['title'] = soup.find('h1', class_='post-title single-post-title entry-title').get_text().replace('\n', '')
        body = soup.find('div', id='penci-post-entry-inner').get_text().replace('\n', '')
        # paras = body.find_all('div', id_='inner-post-entry entry-content jpibfi_container')
        article['body'] = body
        article['words_count'] = len(article['body'].split())
        return article
    except Exception as e:
        print(f"Error scraping data from {url}: {e}")
        return None

# <h1 class="post-title single-post-title entry-title">How To Visit Dhigurah Island: Budget Paradise In Maldives</h1>
# <div class="inner-post-entry entry-content jpibfi_container" id="penci-post-entry-inner" data-content-ads-inserted="true" data-slot-rendered-content="true">…</div>

def scrape_articles(url_base, pages, min_articles):
    articles_data = []
    articles_count = 0
    for i in range(1, pages + 1):
        url = f"{url_base}/p/{i}"
        articles_links = scrape_articlesLinks(url)
        # links = check_links(articles_links)
        for link in articles_links:
            data = article_data(link)
            if data and data.get('words_count', 0) > 500:
                print(f"Article {articles_count + 1} - {data['title']} - {data['words_count']} words")
                articles_data.append({'index': articles_count + 1, 'label': 'Travel', **data})
                articles_count += 1
                if articles_count >= min_articles:
                    return articles_data
    return articles_data


def json_format(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")


def csv_format(data, filename):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'body', 'words_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in data:
                writer.writerow({'title': article['title'], 'body': article['body'], 'words_count': article['words_count']})
            print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")


travel_url_base = 'https://theworldtravelguy.com/blog'
pages = 5  # Number of pages to scrape
min_articles = 15  # Minimum number of articles to scrape
articles_data = scrape_articles(travel_url_base, pages, min_articles)
os.makedirs("articles", exist_ok=True)
# Save to JSON file
output_file_json = 'Articles/travel.json'
json_format(articles_data, output_file_json)
# Save to CSV file
output_file_csv = 'articles/travel.csv'
csv_format(articles_data, output_file_csv)
