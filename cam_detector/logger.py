import logging

log = logging.getLogger(__name__)
handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter(
#                         '%(asctime)-15s [%(levelname)s] %(message)s',
#                         '%Y-%m-%d %H:%M:%S'))

handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

log.addHandler(handler)
log.setLevel(logging.INFO)


if __name__ == '__main__':
    log.info(f'---------------- data info ----------------')
    log.info(f'▶︎ Total: 9999')

    log.info(f'         pos         neg')
    log.info(f' Train: 9999, 9999')
    log.info(f' Valid: 9999, 9999')
    log.info(f'  Test: 9999, 9999')
