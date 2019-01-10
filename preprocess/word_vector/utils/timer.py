from datetime import datetime, timedelta


class Timer:
    def __init__(self):
        self.now = None

    def start(self):
        self.now = datetime.now()

    def restart(self):
        print(datetime.now() - self.now)
        self.now = datetime.now()

    def print_time(self):
        print(datetime.now() - self.now)


def runtime(last=None):
    now = datetime.now()
    if last is None:
        return now

    else:
        run_time = now-last
        print(run_time)

        return now


def runtime_with_budget(last=None):
    now = datetime.now()
    if last is None:
        return now

    else:
        run_time = now-last
        print(run_time)
        if run_time > timedelta(hours=1):
            print('Warning ------ out of time budget')

        return now
