from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import signal

try:
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.debugger_address = "127.0.0.1:9222"

    # Path to ChromeDriver
    service = Service("/usr/bin/chromedriver")  # Adjust path as needed

    # Connect to Chrome
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Example: Open a URL
    # driver.get("https://example.com")
    print("Page title is:", driver.title)

finally:
    # Ensure proper termination of the service
    if service and service.process:
        try:
            os.kill(service.process.pid, signal.SIGTERM)  # Forcefully terminate
        except PermissionError:
            print("Insufficient permissions to terminate the driver process.")
        except Exception as e:
            print(f"Error terminating driver process: {e}")
