import scrapy
from selenium import webdriver
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import time
import cv2
import numpy as np
import csv
import random
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from multiprocessing import Pool
import argparse



class WebpageScreenshotSpider(CrawlSpider):
    name = "screenshot_spider"

    custom_settings = { 
        'DEPTH_LIMIT' :3,'LOG_LEVEL' : 'ERROR' , 'ROBOTSTXT_OBEY': True
    }


    # Define a rule for the links to follow
    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )


    link_counter = 0
    max_links = 10000

    def __init__(self, image_number, urls, *args, **kwargs):
        super(WebpageScreenshotSpider, self).__init__(*args, **kwargs)
        options = Options()
        options.headless = True
        self.driver = webdriver.Chrome(options=options)  # Start Chrome
        self.driver.set_window_size(1920,1080)
        self.image_number= int(image_number)
        self.start_urls = [url.strip() for url in urls.split(',')]
        
    def is_element_in_viewport(self, element):
        viewport = {
            "top": self.driver.execute_script("return window.pageYOffset"),
            "left": self.driver.execute_script("return window.pageXOffset"),
            "width": self.driver.execute_script("return document.documentElement.clientWidth"),
            "height": self.driver.execute_script("return document.documentElement.clientHeight")
        }

        location = element.location
        size = element.size

        return (
            location['y'] >= viewport["top"] and
            location['x'] >= viewport["left"] and
            location['y'] + size['height'] <= viewport["top"] + viewport["height"] and
            location['x'] + size['width'] <= viewport["left"] + viewport["width"] and
            element.is_displayed()  # Check if the element is displayed
        )


    def annotate_viewport(self, viewport_width, viewport_height, clickables):
        # A mapping from element type to class index
        element_type_to_index = {
            'button': 0,
            'heading': 1,
            'link': 2,
            'label': 3,
            'text': 4,
            'image': 5,
            'iframe': 6
        }

        annotations = []
        screenshot_name = f'./wikiset/train/images/image-{self.image_number}.png'
        self.driver.save_screenshot(screenshot_name)
        image = cv2.imread(screenshot_name)

        # Annotate all the different element types
        for element_type, elements in clickables.items():
            for element in elements:
                x, y = element.location['x'], element.location['y']
                width, height = element.size['width'], element.size['height']

                # Calculate relative coordinates and size
                x_center, y_center = (x + width / 2) / viewport_width, (y + height / 2) / viewport_height
                rel_width, rel_height = width / viewport_width, height / viewport_height

                # Get the class index for this element type
                class_index = element_type_to_index[element_type]

                # YOLOv5 annotation format: <class> <x_center> <y_center> <width> <height>
                annotations.append(f"{class_index} {x_center} {y_center} {rel_width} {rel_height}")

                # Draw the box
                cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)

        # Save annotations to a file
        with open(f"./wikiset/train/labels/image-{self.image_number}.txt", "w") as file:
            file.write("\n".join(annotations))

        # Save the image with boxes
        cv2.imwrite(f'./wikiset/train/boxes/image_boxes-{self.image_number}.png', image)


    def parse_item(self, response):
        if self.link_counter >= self.max_links:
            print("DONE")
            exit(0)

        self.driver.get(response.url)
        time.sleep(3)

        # Calculate viewport dimensions
        viewport_width = self.driver.execute_script("return document.documentElement.clientWidth")
        viewport_height = self.driver.execute_script("return document.documentElement.clientHeight")

        # Locate different elements
        elements = self.driver.find_elements(By.XPATH, '//*')
        viewport_elements = [element for element in elements if self.is_element_in_viewport(element)]

        # Group elements by type
        # Update the selectors and logic as needed
        buttons = [e for e in viewport_elements if e.tag_name == "button"]
        headings = [e for e in viewport_elements if e.tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]]
        links = [e for e in viewport_elements if e.tag_name == "a"]
        labels = [e for e in viewport_elements if e.tag_name == "label"]
        texts = [e for e in viewport_elements if e.tag_name == "p"]  # This is a simplification, you might need more complex logic here
        images = [e for e in viewport_elements if e.tag_name in ["img", "svg"]]
        iframes = [e for e in viewport_elements if e.tag_name == "iframe"]

        clickables = {
            'button': buttons,
            'heading': headings,
            'link': links,
            'label': labels,
            'text': texts,
            'image': images,
            'iframe': iframes
        }

        self.link_counter += 1

        # Annotate viewport
        self.annotate_viewport(viewport_width, viewport_height, clickables)

        self.image_number += 1

        def start_requests(self):
            with Pool(processes=10) as pool:  # Number of concurrent processes
                pool.map(self.parse_item, self.start_urls)

    def closed(self, reason):
        self.driver.quit()

        
