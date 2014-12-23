from os.path import dirname, join, abspath


ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
RESOURCES_DIR = join(ROOT_DIR, 'res')

DATASETS = {
    'learn': join(ROOT_DIR, 'res', 'data', 'learn')
}