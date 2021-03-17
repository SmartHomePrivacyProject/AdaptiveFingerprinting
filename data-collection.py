import os
import subprocess
import time
import pandas as pd
import socket
import tldextract
import gc

from tbselenium.tbdriver import TorBrowserDriver
from selenium.common.exceptions import WebDriverException as wde
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import warnings
warnings.filterwarnings('ignore')

# path to tor browser bundle
TBB_PATH = '/home/ubuntu/Documents/webfp/tbb/tor-browser_en-US/'

# path to pcap files
PACP_PATH = '/home/ubuntu/Documents/webfp/output-new/pcap/monitored'
# path for saving screenshots
SCREEN_SHOT = '/home/ubuntu/Documents/webfp/output-new/screen shots/monitored/'

def tor_web_crawler(index, link, ip_address):
    """
    This function is a web crawler for collection of traffic traces and saving those traces to pcap files.
    :param index: current trace of the link
    :param link: webpage address from where traffic is to be collected
    :param ip_address: ip-addres of the machine from which traffic is to be collected
    :param timeout: duration upto which traffic information needs to be collected
    :param pkt_count: number of packets to be collected for a particular trace
    :return:
    """

    # Extracting domain name for saving trace separately
    url = link
    lnk = tldextract.extract(url)
    domain_name = lnk.domain + '.' +lnk.suffix
    # print('Processing trace for domain name crawl : ', domain_name)

    # interface = 'enp0s31f6'
    # interface = 'any'
    interface = 'eth0'
    cap = DesiredCapabilities().FIREFOX
    cap["marionette"] = True  # optional
    # driver = TorBrowserDriver(TBB_PATH)
    try:
        driver = TorBrowserDriver(TBB_PATH)
        # saving the pcapfiles
        PP = PACP_PATH + '/' + domain_name
        # saving the screen shots
        SS = SCREEN_SHOT + '/' + domain_name
        driver.get(url)
    except wde as e:
        print('Browser crashed:')
        print(e)
        print('Trying again in 10 seconds ...')
        time.sleep(10)
        driver = driver
        print('Success!\n')
    except Exception as e:
        raise Exception(e)

    if not os.path.isdir(PP):
        print('Creating directory for saving capture files (pcap) ...')
        os.makedirs(PP)
    else:
        pass

    if not os.path.isdir(SS):
        print('Creating directory for saving screenshots ...')
        os.makedirs(SS)
    else:
        pass

    # command to be executed for capturing the trace
    # command = "sudo tcpdump -i " + str(interface) + " -n host " + str(ip_address) + " -c " + str(pkt_count) + " -w " + PP + "/" + domain_name + "_" + str(index) + ".pcap "
    command = "sudo timeout 120 tcpdump -i " + str(interface) + " -n host " + str(ip_address) + " -w " + PP + "/" + domain_name + "_" + str(index) + ".pcap"
    print('Capture trace ...')
    capture = subprocess.Popen(command, shell=True)
    time.sleep(1)
    capture.wait()
    print('Traffic trace captured and saved successfully.')
    # save the screenshot
    driver.save_screenshot(SS + '/' + domain_name + '-' + str(index) + '.png')
    print('Screen shot of the webpage saved successfully.')
    driver.quit()

if __name__ == '__main__':
    print('Starting to collect traffic trace for the webpages of similar topics...')

    # IP_ADDRESS = '10.63.7.124'
    IP_ADDRESS = '192.168.39.14'
    print('IP-Address : ', IP_ADDRESS)

    # Number of traces to be collected for a partiular link
    traces = 1

    links_path = '../links/keyword-data.xlsx'
    # getting excel file containing links
    links = pd.read_excel(links_path)
    links = links[:40]

    print('Web crawler started ...')
    start_time = pd.Timestamp.now()
    # getting traces for the links
    for j in range(traces):
        print('Batch : %d'%(j+1))
        print('+'*80)
        # for i in range(len(links)):
        for i in range(1):
            print('Trace: %d for domain %s'%(j+1, links.iloc[i][0]))
            start_time_tr = pd.Timestamp.now()
            tor_web_crawler(j+1, links.iloc[i][1], IP_ADDRESS)
            end_time_tr = pd.Timestamp.now() - start_time_tr
            print('Time taken to collect trace: ', end_time_tr)
            gc.collect()
            print('*'*80)

    end_time = pd.Timestamp.now() - start_time
    print('program execution completed.')
    print('Time taken for data collection: ', end_time)
