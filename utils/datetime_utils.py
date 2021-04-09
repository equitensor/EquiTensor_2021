import datetime

def str_to_datetime(input_str):
    datetimeFormat = "%Y-%m-%d %H:%M:%S"
    return datetime.datetime.strptime(input_str,datetimeFormat)

def datetime_to_str(input_datetime):
    return input_datetime.strftime("%Y-%m-%d %H:%M:%S")


# given a start, end time, generate a series of hour
def datetime_range(start, end, delta):
    if isinstance(start, str):
        start = str_to_datetime(start)
    if isinstance(end, str):
        end = str_to_datetime(end)

    current = start
    if not isinstance(delta, datetime.timedelta):
        delta = datetime.timedelta(**delta)
    while current <= end:
        yield current
        current += delta

# compute the total time in hours of period in consideration
# e.g.: return 360 for ('betweem 2017-10-01 00:00:00' and '2017-10-15 23:59:59')
# input: strings. eg: '2017-10-01'
def get_total_hour_range(start_date, end_date):
    datetimeFormat = "%Y-%m-%d %H:%M:%S"
    t1 = start_date+' 00:00:00'
    t2 = end_date+' 23:59:59'
    timedelta = datetime.datetime.strptime(t2, datetimeFormat) - datetime.datetime.strptime(t1,datetimeFormat)
    return (timedelta.days + 1) * 24



def get_total_3hour_range(start_date, end_date):
    datetimeFormat = "%Y-%m-%d %H:%M:%S"
    t1 = start_date+' 00:00:00'
    t2 = end_date+' 23:59:59'
    timedelta = datetime.datetime.strptime(t2, datetimeFormat) - datetime.datetime.strptime(t1,datetimeFormat)
    return (timedelta.days + 1) * 8
