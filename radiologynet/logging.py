from datetime import datetime


def log(*message, **kwargs):
    default_kwargs = {'verbose': True, 'end': '\n'}
    kwargs = {**default_kwargs, **kwargs}
    verbose, end = kwargs['verbose'], kwargs['end']
    if(verbose):
        now = str(datetime.now())
        msg = ''
        for m in message:
            msg = f'{msg} {m}'
        print(f'*** {now} {msg} ***', end=end)
