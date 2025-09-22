# searx_spider.py
import json, re
from scrapy import Spider, Request
from bs4 import BeautifulSoup

class SearxSpider(Spider):
    name = 'searx_spider'
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_DELAY': 0.5,
        'USER_AGENT': 'Mozilla/5.0 (compatible; YourBot/1.0)'
    }

    def __init__(self, urls_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not urls_file:
            raise ValueError('Provide --a urls_file=path')
        self.urls = []
        with open(urls_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    j = json.loads(line)
                    url = j.get('url')
                    if url:
                        self.urls.append({'url': url, 'query': j.get('query'), 'snippet': j.get('snippet')})
                except Exception:
                    continue

    def start_requests(self):
        for item in self.urls:
            yield Request(url=item['url'], callback=self.parse, meta=item)

    def parse(self, response):
        text = ''
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            for s in soup(['script','style','noscript']):
                s.decompose()
            text = soup.get_text(separator=' ')
            text = re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            self.logger.error('parse error %s', e)
        yield {
            'url': response.url,
            'query': response.meta.get('query'),
            'snippet': response.meta.get('snippet'),
            'text': text
        }
