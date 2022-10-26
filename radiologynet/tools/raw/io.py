from typing import Iterable, Union
import typing
import pydicom
import os
import csv
import pandas as pd
import dask.dataframe as dd
import multiprocessing
from radiologynet.logging import log
import numpy as np


def get_path_from_id(disk_label: str, id: int, extension='dcm'):
    """
    Returns the full path to a file according to specific rules.

    The ruleset used for generating the full path is as follows:
    - disk_image is the full path to the folder where all of the files
        are located.
    - the next part of the path consists of the id, excluding the last three
        digits.
    - a delimiter is placed before the next step
    - the full ID is placed here, and afterwards a dot ('.') followed
        by the extension.

    IMPORTANT: This function does not check if the generated
    file path is actually valid and that the file exists on
    filesystem.

    Args:
        disk_label: location of the folder where all of the files \
            are located (Scirocco, Tramontana)

        id (`int`): unique identifier of the DICOM object.

        extension (str, optional): extension to be appended to the \
            end of the fullpath. Useful in cases where images might be
            saved as something other than DCM format (e.g. PNG).
            Defaults to `'dcm'`.

    Returns:
        A string, the full path to the file.
    """
    path = os.path.join(
        disk_label,
        str(id)[0:-3],
        f'{id}.{extension}'
    )
    return path


def get_dicom_from_path(path: str, stop_before_pixels=False):
    """Get the DICOM object from the specified path.

    Args:
        path (str): Full path to the DCM object.

        stop_before_pixels (bool, optional): If `False`, returns \
            the whole DICOM file, otherwise returns everything but
            `Pixel Data` and subsequent data (kindly refer
            to the `pydicom.dcmread` docs). Defaults to True.

    Raises:
        FileNotFoundError: No file was found on the provided path.
        Exception: A general exception occured.

    Returns:
        An object (FileDataset | DicomDir): the read DCM object.
    """
    try:
        dicom_data = pydicom.dcmread(
            path, stop_before_pixels=stop_before_pixels)
    except FileNotFoundError:
        raise FileNotFoundError('Filepath provided is not a valid DICOM file.')
    except Exception as e:
        raise Exception(
            ('Failed to read DICOM file from path ' +
             path +
             ' due to error: ' + str(e))
        )

    return dicom_data


def get_dicom_by_id(
    foldername: str,
    dcm_id: int,
    stop_before_pixels: bool = False
):
    path = get_path_from_id(foldername, dcm_id)
    return get_dicom_from_path(path, stop_before_pixels=stop_before_pixels)


def get_dicoms_from_folder(
        folderpath: str,
        stop_before_pixels: bool = True,
        return_id: bool = True,
        return_dcm: bool = True
):
    """Get DICOM files from folder.

    Args:
        folderpath (str): Location of the folder where DICOM \
            files are located.

        stop_before_pixels (bool, optional): If `False`, returns \
            the whole DICOM file, otherwise returns everything but
            `Pixel Data` and subsequent data (kindly refer
            to the `pydicom.dcmread` docs). Defaults to True.

        return_id (bool, optional): If `True`, a part of the return \
            value is the DICOM ID. Defaults to `True`.

        return_dcm (bool, optional): If `True`, a part of the return \
            is the DICOM file and the read tags. Defaults to `True`.

    Yields:
        * If both `return_id` and `return_dcm` are `True`,then `(dcm_id, dcm)`.
        * If only `return_id` is True, then `dcm_id`.
        * If only `return_dcm` is True, then `dcm_dcm`.
        * if both are `False`, asn `AssertionError` is thrown.

    """
    assert folderpath is not None, 'folderpath cannot be None!'
    assert not (
        return_dcm is False and return_id is False
    ), 'At least one value should be returned!'

    for subfolderpath, _, filenames in os.walk(folderpath):
        for filename in filenames:
            if filename.endswith('.dcm'):
                fullpath_to_dcm = os.path.join(subfolderpath, filename)
                dcm_id = int(filename[0:-4])
                dcm = get_dicom_from_path(
                    fullpath_to_dcm,
                    stop_before_pixels=stop_before_pixels
                )

                if return_dcm is True and return_id is True:
                    yield dcm_id, dcm
                elif return_dcm is True:
                    yield dcm
                else:
                    yield dcm_id


def convert_single_dicom_to_dataframe(dcm, dcm_id: int):
    tags = dcm.dir()
    values = {tag: dcm.get(tag) for tag in tags}
    values['id'] = dcm_id
    dfn = pd.DataFrame.from_records([values], index='id')
    return dfn


def convert_dicoms_to_dataframe(
        folderpath: str,
        dcm_ids: Iterable[int] = None
):
    """Convert multiple DICOM objects into a pandas dataframe.

    Args:
        folderpath (str): Path to where the DICOM objects are located.

        dcm_ids (List[int], optional): Use if only specific DICOM IDs \
            are required. Pass an array of DICOM IDs which should to be \
            converted. Please note that `folderpath` \
            and each DICOM ID is formatted into a filepath according to \
            `get_path_from_id()` method rules. \
            If this is `None`, then all of the DICOM\
            files in the folder `folderpath` are converted. \
            Defaults to None.

    Returns:
        Dataframe: a pandas dataframe containing the merged DICOM files.
    """
    if (dcm_ids is not None):
        # get all of the specified DCM files (according to dcm_ids)
        # expressed as a generator of (dcm_id, dcm_file) pairs.
        to_convert = (
            (dcm_id, get_dicom_by_id(foldername=folderpath,
             dcm_id=dcm_id, stop_before_pixels=True))
            for dcm_id in dcm_ids
        )

    else:
        # gather all of the DICOM files in the specified folder.
        to_convert = get_dicoms_from_folder(
            folderpath,
            stop_before_pixels=True
        )

    df: pd.DataFrame = pd.DataFrame()
    for dcm_id, dcm in to_convert:
        try:
            dfn = convert_single_dicom_to_dataframe(dcm, dcm_id)
            df = pd.concat([df, dfn], sort=False)
        except Exception as e:
            continue
    return df


def get_unique_tags_from_dicoms(dicoms: Iterable):
    unique_tags = set()

    for dcm in dicoms:
        unique_tags.update(dcm.dir())

    return unique_tags


def export_dicom_metadata_from_folder_to_csv(
    entry_name: str,
    path_to_entry: str,
    save_to: str,
    verbose: bool = False,
    NaN_replacement=None,
    force_rewrite: bool = False
):
    path_to_csv = os.path.join(save_to, entry_name + '.csv')
    # check if CSV already exists, and if it does
    # then skip this folder entirely
    if force_rewrite is False and os.path.isfile(path_to_csv):
        log('!! CSV found, skipping folder:' + entry_name, verbose=verbose)
        return

    log('Exporting folder:' + entry_name + '...', verbose=verbose)

    # if the CSV has not already been generated, then generate one
    # and place it in metadata/ subfolder
    try:
        df = convert_dicoms_to_dataframe(path_to_entry)
    except Exception as e:
        log(e, verbose=verbose)
        return

    if NaN_replacement is not None:
        df = df.fillna(NaN_replacement)

    df.to_csv(path_to_csv)
    log('Successfully exported folder:' + entry_name + '.', verbose=verbose)


def export_dicom_metadata_from_folders_to_csv(
    from_folders: Iterable[os.DirEntry],
    save_to: str,
    verbose=False,
    NaN_replacement=None,
    nprocesses: int = None,
    force_rewrite: bool = False
):
    """Export all DCM metadata \
    (found in specified folders) to CSV files.

    For each folder in `from_folders`, get all of its DICOM files. \
    Each DICOM file will be loaded as an entry in a pandas dataframe and \
    then exported to CSV. The CSV will have the same filename as the folder \
    it originated from. For example, if the folder is named `14305` and it \
    has three DICOM files: `[14305000.dcm, 14305001.dcm, 14305002.dcm]`, \
    then the exported CSV file will be named `14305.csv` \
    and will contain three rows, \
    one row for each of the files.

    Args:
        from_folders (Iterable[os.DirEntry]): an iterable of folders \
            where DICOM files can be found.

        save_to (str): The location where to save the CSV files. \
            Make sure this path ends in the path delimiter.

        verbose (bool, optional): If True, prints out export progress. \
            Defaults to False.

        NaN_replacement (Any, optional): If not None, this value \
            will be used to replace all of the NaN values \
            (all of the empty DICOM tags) before exporting to CSV.\
            Defaults to None.

        nprocesses (int, optional): For multiprocessing, how many \
            processes to create & use for exporting. Defaults to None.

        force_rewrite (bool, optiona): If there already exists a CSV \
            with the same name as the one which should be exported, \
            then, if this is `True`, the CSV file will be re-written. \
            Defaults to `False`.

    """
    log('Begin export to CSV :: saving to' + save_to, verbose=verbose)

    with multiprocessing.Pool(nprocesses) as pool:
        args = []
        for entry in from_folders:
            # skip non-directories and empty directories
            if entry.is_dir() and os.listdir(entry.path):
                args.append(
                    (
                        entry.name,
                        entry.path,
                        save_to,
                        verbose,
                        NaN_replacement,
                        force_rewrite
                    )
                )
        pool.starmap(export_dicom_metadata_from_folder_to_csv, args)

    log('EXPORT FINISHED', verbose=verbose)


def get_csv_files_as_dask_dataframe(
    csv_filepaths,
    NaN_replacement=None,
    return_computed: bool = True,
    verbose: bool = False,
    engine: str = 'python',
    on_bad_lines: str = 'skip',
    set_id_as_index: bool = True,
    header: typing.Union[int, typing.List[int], str] = 'infer'
) -> Union[dd.DataFrame, pd.DataFrame]:
    """Load all the provided CSV files and concatenate them
    into a single Dask dataframe.

    Args:
        csv_filepaths (Any): a selector Dask.DataFrame.read_csv
            is able to parse.

        NaN_replacement (Any, optional): If not None, this value
            will be used to replace all of the NaN values. This
            does nothing if `return_computed` is `False`.
            Defaults to None.

        return_computed (bool, optional): If False, returns a
            Dask task which has to be computed before further usage.
            Defaults to True.

        verbose (bool, optional): If True, prints out helpful logs.
            Defaults to False.

        engine (str, optional): Which engine to use when reading CSVs.
            Read pandas documentation for details.
            Defaults to 'python'.

        on_bad_lines (str, optional): What to do with a CSV line
            cannot be parsed.
            Read pandas documentation for details.
            Defaults to 'skip'.

        set_id_as_index (bool, optional): whether to automatically
            set the column named 'id' as the dataframe index.
            Prerequisite: the CSVs must contain a header and
            must have a column named 'id' which is of type `int32`.
            Defaults to True.

        header (typing.Union[int, typing.List[int], str], optional):
            Values for CSV header.
            Read pandas documentation for details.
            Defaults to 'infer'.

    Returns:
        Dask Dataframe or Dask Task: Returns either
            a Dask DataFrame or a Dask Task, depending
            on the arguments that were given.
    """
    log('Began reading CSVs.', verbose=verbose)
    df: pd.DataFrame = dd.read_csv(
        csv_filepaths,
        dtype=object,
        blocksize=1e8,
        on_bad_lines=on_bad_lines,
        engine=engine,
        header=header
    )
    log(' Finished reading CSVs', verbose=verbose)

    if (return_computed is False):
        return df

    # if return_computed is True -> compute the dataframe before
    # returning its value
    log(' Computing the dask task...', verbose=verbose)
    df: pd.DataFrame = df.compute()
    log(' Task computed!', verbose=verbose)

    if (set_id_as_index is True):
        log(' Casting id to int32...', verbose=verbose)
        df = df.astype({'id': 'int32'})
        log(' Setting index on the DataFrame...', verbose=verbose)
        df = df.set_index('id')

    if (NaN_replacement is not None):
        df = df.fillna(NaN_replacement)

    log(' All done! Returning the value...', verbose=verbose)

    return df


def drop_rows_and_cols_with_nan_values(
    df: pd.DataFrame,
    how: str = 'all',
    verbose: bool = False
) -> pd.DataFrame:
    """Drop rows and columns which contain NaN values form a dataframe.

    Args:
        df (pd.DataFrame): DataFrame to drop from.
        how (str, optional): If `'all'`, drops \
            rows/cols where ALL values are NaN. \
            If `'any'`, drops rows/cols where there is at least \
            one NaN value. Defaults to 'all'.

    Returns:
        DataFrame: the resulting dataframe.
    """
    log(f'Dropping where NaN=`{how}` from dataframe...', verbose=verbose)
    df.dropna(axis='columns', how=how, inplace=True)
    df.dropna(axis='index', how=how, inplace=True)
    return df


def export_iterable_to_csv(
    iterable: Iterable,
    as_one_dimensional=False,
    to_filename='exported.csv',
    delimiter=',',
):
    """Export an array (or any iterable) to a CSV file.

    Args:
        iterable (Iterable): The iterable which should be written to CSV.

        as_one_dimensional (bool, optional): should the iterable in question \
            be treated as it's one dimensional. Set this to `True` \
            when writing 1D arrays to CSV file. Defaults to `False`.

        to_filename (str, optional): The full name (file extension included) \
            of the newly created file. Defaults to `'exported.csv'`.

        delimiter (str, optional): CSV delimiter.. Defaults to `','`.
    """

    file = open(to_filename, 'w')

    csvwriter = csv.writer(file, delimiter=delimiter)

    for row in iterable:
        content_to_write = [row] if as_one_dimensional is True else row
        csvwriter.writerow(content_to_write)

    file.close()


def load_csv_metadata_and_export_to_sql(
    csv_filepaths,
    sql_uri: str = 'sqlite:///issa_metadata.sql',
    drop_NaN_cols: bool = True,
    NaN_replacement='',
    verbose: bool = False,
    table_name: str = 'MetaInformation',
    npartitions=100
):
    """Perform a full cycle: load CSVs, drop & clean NaN
        values, and then export to SQL.

    Args:
        csv_filepaths (Any): Where to find CSVs from which to form a
            dataframe. Please make sure this input is compatible with
            what Dask/Pandas expects as input for `read_csv`.
        sql_uri (str, optional): Where to export.
            Defaults to `'sqlite:///issa_metadata'`.
        drop_NaN_cols (bool, optional): If `True`, columns with all-Nan
            values will be dropped from the dataframe. Defaults to `True`.
        NaN_replacement (str, optional): What to fill NaN values with.
            Defaults to `''`.
        verbose (bool, optional): Whether to print out helpful messages.
            Defaults to `False`.
        table_name (str, optional): The dataframe's content will be
            exported to an SQL table with this name.
            Defaults to `'MetaInformation'`.
        npartitions (int, optional): for Dask magic. Defaults to 100.
    """
    df = get_csv_files_as_dask_dataframe(
        csv_filepaths,
        NaN_replacement=None,
        return_computed=True,
        verbose=verbose
    )
    if (drop_NaN_cols):
        df = drop_rows_and_cols_with_nan_values(df, verbose=verbose)
    df = dd.from_pandas(df, npartitions=npartitions)

    export_dataframe_to_sql(
        df,
        sql_uri=sql_uri,
        NaN_replacement=NaN_replacement,
        verbose=verbose,
        table_name=table_name
    )


def export_dataframe_to_sql(
    df: dd.DataFrame,
    sql_uri: str = 'sqlite:///issa_metadata',
    NaN_replacement=None,
    verbose: bool = False,
    table_name: str = 'MetaInformation',
    if_exists: str = 'replace',
    chunksize: int = None
):
    """Export a dask dataframe to SQL.

    Args:
        df (dd.DataFrame): Dataframe to export.
        sql_uri (str, optional): Where to export.\
            Defaults to `'sqlite:///issa_metadata'`.
        NaN_replacement (str, optional): What to fill NaN values with.\
            Defaults to `''`.
        verbose (bool, optional): Whether to print out helpful messages.\
            Defaults to `False`.
        table_name (str, optional): The dataframe's content will be \
            exported to an SQL table with this name. \
            Defaults to `'MetaInformation'`.
        if_exists (str, optional). Either `'fail'`, `'append'` or
            `'replace'`. Refer to the dask/pandas docs for details.
            Defaults to `'replace'`.
        chunksize (int, optional). The number of rows which to write
            int chunks. Refer to dask/pandas docs for details. None means
            that everything will be written at once.
            Defaults to None.
    """
    if (NaN_replacement is not None):
        log(f'Filling NaN Values with "{NaN_replacement}"', verbose=verbose)
        df = df.fillna(NaN_replacement)

    log(f'Finally, exporting to: "{sql_uri}"', verbose=verbose)

    df.to_sql(
        table_name,
        sql_uri,
        chunksize=chunksize,
        if_exists=if_exists,
        # parallel=True,
    )
    log('Export finished', verbose=verbose)


def parse_text_db(
    start_char_i: int,
    text_db_path: str,
    headers_path: str,
    save_dir: str,
    delim: str = ';',
    csv_max_nrow=1000,
    verbose: bool = False
):
    log(f'Starting to parse text_db at {text_db_path}')
    file = open(headers_path)
    headers = file.readline()
    file.close()
    file = open(text_db_path)
    rpt = file.readlines()
    file.close()

    # each column has a fixed number of characters it can take up
    # so this is to count how many chars does a single
    # column take up
    header_num = len(headers.split(';'))
    line = rpt[1]
    cnt = np.zeros(header_num)
    i = 0
    for char in line:
        cnt[i] += 1
        if (char.isspace()):
            i += 1

    parsed = []
    # replace all newlines with spaces
    lines = ' '.join(rpt[2:])
    lines = lines.replace('\r\n', ' ')

    # at which character index to start counting
    # rows
    k = start_char_i
    try:
        while (k < len(lines)):
            parsed_line = []
            log(f'{k} <<<----- NEWLINE', verbose=verbose)
            for i in range(len(cnt)):
                start = k
                end = int(cnt[i]) if i + \
                    1 < len(cnt) else lines[start:].index('\n') + 1
                end += start
                parsed_line.append(f'"{lines[start:end].strip()}"')
                k = end
            # log(f'{k} --->>> LINE FINISHED, appending to CSV',
            #     verbose=verbose)
            parsed.append(parsed_line)

            if (len(parsed) >= csv_max_nrow):
                save_text_db_parsed(
                    parsed, headers=headers, save_dir=save_dir,
                    k=k, delim=delim, verbose=verbose)
                parsed = []
    except Exception as e:
        log(f'Interrupted: {e}, but saving what was parsed so far...')
    finally:
        save_text_db_parsed(
            parsed, headers=headers, save_dir=save_dir,
            k=k, delim=delim, verbose=verbose)


def save_text_db_parsed(
    parsed: Iterable[str],
    headers: str,
    save_dir: str,
    k: int,
    delim: str,
    verbose: bool = False
):
    parsed = [delim.join(line) for line in parsed]
    log(f'saving to {k}.csv', verbose=verbose)
    parsed_csv = open(f'{save_dir}{k}.csv', mode='w')
    parsed_csv.writelines([headers, '\n'.join(parsed)])
    parsed_csv.close()
