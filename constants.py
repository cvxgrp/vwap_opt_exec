__all__ = ['SAVEFOLDER', 'PROCESSEDDATAFOLDER', 'RAWDATAFOLDER', 'GRAPHICSFOLDER', 
        'NUM_INTERVALS', 'NUM_INTERVALS_IN_FILE', 'ALL_SYMBOLS', 'ALL_DAYS', 
        'DEBUG', 'logger']

import inspect, os
CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 

SAVEFOLDER = CUR_DIR+"/../Data/Pickles/"
PROCESSEDDATAFOLDER = CUR_DIR+"/../Data/DailyProfiles/"
RAWDATAFOLDER = CUR_DIR+"/../Data/RawData/"
GRAPHICSFOLDER = CUR_DIR+"/../Graphics/"
NUM_INTERVALS = 390
NUM_INTERVALS_IN_FILE = 392
ALL_SYMBOLS = [
    "MMM", "AXP", "T", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
    "GE", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT", 
    "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "DIS", "AA", "BAC", 
    "HPQ"
    ]
ALL_DAYS = [
    '20120924', '20120925', '20120926', '20120927', '20120928', '20121001', 
    '20121002', '20121003', '20121004', '20121005', '20121008', '20121009', 
    '20121010', '20121011', '20121012', '20121015', '20121016', '20121017', 
    '20121018', '20121019', '20121022', '20121023', '20121024', '20121025',
    '20121026', '20121031', '20121101', '20121102', '20121105', '20121106', 
    '20121107', '20121108', '20121109', '20121112', '20121113', '20121114', 
    '20121115', '20121116', '20121119', '20121120', '20121121', '20121126', 
    '20121127', '20121128', '20121129', '20121130', '20121203', '20121204', 
    '20121205', '20121206', '20121207', '20121210', '20121211', '20121212', 
    '20121213', '20121214', '20121217', '20121218', '20121219', '20121220']

DEBUG = True

# logs - export only the logger
import logging
reload(logging) #for iPython interactive usage
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler(CUR_DIR+'/../code_errors_log_.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
