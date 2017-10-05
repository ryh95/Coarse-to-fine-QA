import zipfile

import os
from urllib import urlretrieve, reporthook

from tqdm import tqdm


def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check."""
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix,filename), reporthook=reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix,filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully loaded".format(filename))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename

if __name__ == '__main__':
    glove_base_url = "http://nlp.stanford.edu/data/"
    glove_filename = "glove.6B.zip"
    prefix = os.path.join("../","data", "dwr")

    print("Storing datasets in {}".format(prefix))

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    glove_zip = maybe_download(glove_base_url, glove_filename, prefix, 862182613L)
    glove_zip_ref = zipfile.ZipFile(os.path.join(prefix, glove_filename), 'r')

    glove_zip_ref.extractall(prefix)
    glove_zip_ref.close()
