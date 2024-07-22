import sys 
import os
import numpy as np
import logging 
from termcolor import colored
import cv2

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


def getchar():
    print('press enter to continue:')
    a = input('').split(" ")[0]
    print(a)
    

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False    

class Colors(object):
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'
    class fg: 
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
    class bg: 
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'

class Printer(object):
    @staticmethod
    def red(*args, **kwargs):
        print(Colors.fg.red, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def green(*args, **kwargs):
        print(Colors.fg.green, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def blue(*args, **kwargs):
        print(Colors.fg.blue, *args, **kwargs)
        print(Colors.reset, end="")        
        
    @staticmethod
    def cyan(*args, **kwargs):
        print(Colors.fg.cyan, *args, **kwargs)
        print(Colors.reset, end="")             
        
    @staticmethod
    def orange(*args, **kwargs):
        print(Colors.fg.orange, *args, **kwargs)
        print(Colors.reset, end="")     
        
    @staticmethod
    def purple(*args, **kwargs):
        print(Colors.fg.purple, *args, **kwargs)
        print(Colors.reset, end="")  
        
    @staticmethod
    def yellow(*args, **kwargs):
        print(Colors.fg.yellow, *args, **kwargs)
        print(Colors.reset, end="")                                   

    @staticmethod
    def error(*args, **kwargs):
        print(Colors.fg.red, *args, **kwargs, file=sys.stderr)
        print(Colors.reset, end="")        


# test class with termcolor
class Printer_old(object):
    @staticmethod
    def red(input):
        print(colored(input,'red'))

    @staticmethod
    def green(input):
        print(colored(input,'green'))           


# return a random RGB color tuple 
def random_color():
    color = tuple(np.random.randint(0,255,3).tolist())    
    return color 


# for logging to multiple files, streams, etc. 
class Logging(object):
    time_log_formatter = logging.Formatter('%(levelname)s[%(asctime)s] %(message)s')
    notime_log_formatter = logging.Formatter('%(levelname)s %(message)s')
    simple_log_formatter = logging.Formatter('%(message)s')
    thread_log_formatter = logging.Formatter('%(levelname)s] (%(threadName)-10s) %(message)s')
    
    @staticmethod
    def setup_logger(name, level=logging.INFO, formatter=time_log_formatter): # to sys.stderr
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.StreamHandler()        
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger
        
    @staticmethod
    def setup_file_logger(name, log_file, level=logging.INFO, mode='+w', formatter=time_log_formatter): # to file 
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.FileHandler(log_file, mode=mode)        
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger
    
    @staticmethod
    def setup_socket_logger(name, host, port, level=logging.INFO, formatter=time_log_formatter): # TCP-IP
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.SocketHandler(host, port)        
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger    
    
    @staticmethod
    def setup_udp_logger(name, host, port, level=logging.INFO, formatter=time_log_formatter): # UDP
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.DatagramHandler(host, port)        
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger        


def import_from(module, name, method=None):
    try:      
        imported_module = __import__(module, fromlist=[name])
        imported_name = getattr(imported_module, name)
        if method is None: 
            return imported_name
        else:
            return getattr(imported_name, method) 
    except: 
        if method is not None: 
            name = name + '.' + method 
        Printer.orange('WARNING: cannot import ' + name + ' from ' + module + ', check the file TROUBLESHOOTING.md')    
        return None   
    
    
def print_options(opt, opt_name='OPTIONS'):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    print_notification(content_list, opt_name)    
    
def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(('---------------------- {0} ----------------------'.format(notifi_type)))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')    
    
def get_opencv_version():
    opencv_major =  int(cv2.__version__.split('.')[0])
    opencv_minor =  int(cv2.__version__.split('.')[1])    
    opencv_build = int(cv2.__version__.split('.')[2])    
    return (opencv_major, opencv_minor, opencv_build)

def is_opencv_version_greater_equal(a, b, c):
    opencv_version = get_opencv_version()
    return opencv_version[0]*1000 + opencv_version[1]*100 + opencv_version[2] >= a*1000 + b*100 + c
