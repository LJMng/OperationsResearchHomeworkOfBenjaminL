'''
Created on 2019年12月6日

@author: Benjamin_L
'''

import time

class Timer(object):

    def __init__(self):
        self.start_time = None
        self.total_time = None
        
    def start(self):
        if self.start_time is not None:
            start_time_str = time.strftime('%H:%M:%S', time.localtime(self.start_time))
            raise Exception('Timer already has been started at %s !' % start_time_str)
        
        self.start_time = time.time()
        self.total_time = 0
        
    def stop(self):
        """
        Stop the time recording and return the time record in second(s)
        """
        if self.start_time is None:
            raise Exception('Timer has not been started yet!')
        
        stop = time.time()
        self.total_time = stop - self.start_time
        self.start_time = None
        return self.total_time
    
    def get_total_time(self):
        """
        Get the time record in second(s).
        """
        return self.total_time
    
    def get_total_time_minute(self):
        """
        Get the time record in minute(s).
        """
        return self.get_total_time() / 60

    
def auto_transfer_units(time, current_unit='second'):
    """
    Transfer the given time into better recognizable format/unit.
    millisecond -> second -> minute -> hour -> day -> week
    
    Parameters
    ----------
        time: float
            the time value.
        current_unit: str
            time unit of the argument 'time', 'second' by default.
    
    return time, unit
    """
    recursion = False
    if current_unit is 'millisecond':
        if time > 1000:
            time /= 1000
            to_unit = 'second'
            recursion = True
    elif current_unit is 'second':
        if time > 60:
            time /= 60
            to_unit = 'minute'
            recursion = True
    elif current_unit is 'minute':
        if time > 60:
            time /= 60
            to_unit = 'hour'
            recursion = True
    elif current_unit is 'hour':
        if time > 24:
            time /= 24
            to_unit = 'day'
            recursion = True
    elif current_unit is 'day':
        if time > 7:
            time /= 7
            to_unit = 'week'
            recursion = True
    return auto_transfer_units(time, to_unit) if recursion else (time, current_unit)
    
            
if __name__ == '__main__':
    timer = Timer()
    timer.start()
    
    time.sleep(2)
    
    timer.stop()
    
    print(timer.get_total_time())