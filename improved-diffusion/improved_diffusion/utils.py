import pdb


def point_debug(args):
    if getattr(args, 'debug', False):
        pdb.set_trace()
