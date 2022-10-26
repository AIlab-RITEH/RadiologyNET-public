from typing import Iterable, Union
import typing
import sklearn.preprocessing as prep
import sklearn.decomposition as decomp
import sklearn.manifold as manifold
import pandas as pd
import numpy as np
from collections import namedtuple
import re
from radiologynet.logging import log


class DicomTagFeatureExtractor:
    def __init__(self, tags: pd.DataFrame = None) -> None:
        """Create a feature extractor for DICOM tags.

        WARNING!! This WILL NOT copy the dataframe given as argument.
        Be careful and keep track of dataframe references!

        Args:
            tags (pd.DataFrame, optional): Dataframe containg the \
            DICOM tags. Defaults to None.
        """
        self.tags = tags

    def __str__(self) -> str:
        return f'<DicomTagFeatureExtractor>\n:: Tags:\n{self.tags}'

    def _check_arg_tags(self, tags: pd.DataFrame):
        """Check if there are any tags to work with.

        Args:
            tags (pd.DataFrame): tags to work with.

        Raises:
            AssertionError: if there are no tags to work with.

        Returns:
            DataFrame: tags to work with.
        """
        if (tags is None):
            # if no tags provided, search for tags which
            # were passed through constructor
            tags = self.tags

        assert tags is not None,\
            ('No tags provided, please provide tags either as'
                'argument or through the `tags` class attribute!')
        return tags

# region FILLING_BODY_PART_EXAMINED

    def parse_BodyPartExamined_from_regex(
        self,
        to_value: str,
        regexpr: str,
        from_value: str = '',
        tags: pd.DataFrame = None,
        attribute_to_parse_from: str = "StudyDescription"
    ):
        tags['BodyPartExaminedFixed'] = np.where(
            np.logical_and(
                tags["BodyPartExaminedFixed"] == from_value,
                tags[attribute_to_parse_from].str.contains(
                    regexpr, regex=True, flags=re.IGNORECASE)
            ),
            to_value,
            tags['BodyPartExaminedFixed']
        )
        return tags

    def parse_BodyPartExamined_replace(
        self,
        to_value: str,
        from_value: str,
        tags: pd.DataFrame = None
    ):
        tags['BodyPartExaminedFixed'] = np.where(
            tags["BodyPartExaminedFixed"] == from_value,
            to_value,
            tags['BodyPartExaminedFixed']
        )
        return tags

    def fix_BodyPartExamined(
        self,
        tags: pd.DataFrame = None,
        inplace: bool = False,
        drop_others: bool = False,
    ):
        tags = self._check_arg_tags(tags)
        tags['BodyPartExaminedFixed'] = tags['BodyPartExamined']
        RegexReplacement = namedtuple(
            'RegexReplacement', 'to_val regexpr from_val',
            defaults=['', '', ''])
        ValueReplacement = namedtuple('ValueReplacement', 'from_val to_val')

        regexes: Iterable[RegexReplacement] = [
            RegexReplacement(
                'WRIST', 'wrist|rucni'),
            RegexReplacement('FINGER', 'finger|prst'),
            RegexReplacement(
                'ARM', ('forearm|upper extrem|gornj(?:i|ih|eg) ekstrem|'
                        'nadlaktic|podlaktic|humerus|ruk(?:a|e|u)')),
            RegexReplacement('HAND', 'saka|sake|hand'),
            RegexReplacement('ELBOW', 'lakat|elbow'),
            RegexReplacement(
                'ARM', ('forearm|upper extrem|gornj(?:i|ih|eg) ekstrem|'
                        'nadlaktic|podlaktic|humerus|ruk(?:a|e|u)'),
                from_val='EXTREMITY'),
            # RegexReplacement('TMJ', 'stenvers|tm zglob'),
            RegexReplacement('SKULL', 'stenvers|tm zglob'),
            # RegexReplacement('JAW', 'mandible'),
            RegexReplacement('SKULL', 'mandible'),
            # RegexReplacement('ORBIT', 'orbita'),
            RegexReplacement('SKULL', 'orbita'),
            RegexReplacement('SKULL', 'skull|schädel'),
            RegexReplacement(
                'HEAD', 'glav(?:a|e)|sinus|nasal|head|moz(?:ga|ak)'),
            RegexReplacement(
                'HEAD', 'glav(?:a|e)|sinus|nasal|head|moz(?:ga|ak)',
                from_val='SPINE'
            ),
            # humerus --> bone in the arm
            # gastroscopy => imaging of the stomach etc.
            RegexReplacement(
                'GITRACT',
                'ercp|gastro|jednjak|kolonografija',
                from_val='ABDOMEN'),
            RegexReplacement(
                'GITRACT',
                'ercp|gastro|jednjak|kolonografija',
            ),
            # RegexReplacement('CEREBRALA', 'neuro glue'),
            # neuro glue is actually tied to inspections of the
            # gastrointestinal tract
            # RegexReplacement('GITRACT',
            #                  ('neuro glue')),
            RegexReplacement('HEEL', 'calcaneus|heel|petn(?:a|e)|peta'),
            RegexReplacement('ANKLE', 'glezanj|gleznja|ankle|skocni'),
            RegexReplacement('KNEE', 'knee|koljen'),
            RegexReplacement('KNEE', 'knee|koljen', from_val='EXTREMITY'),
            RegexReplacement('FOOT', 'stopal|foot'),
            RegexReplacement('FOOT', 'stopal|foot', from_val='EXTREMITY'),
            # Calcaneus -- bone in the heel of the foot
            # try to avoid extremity, because it's really not precise enough
            # RegexReplacement(
            #     'EXTREMITY',
            #     'e(?:x|ks)trem(?:i|e)t(?:et|y|ies)'
            # ),
            RegexReplacement(
                'LEG', ('natkoljenica|potkoljenica|lower (?:leg|limb|extrem)|'
                        'nog(?:a|e|u)|femoral(?:|ne) art|'
                        'do(?:|lj)nj(?:i|ih|eg) ekstrem')),
            RegexReplacement(
                'LEG', ('natkoljenica|potkoljenica|lower (?:leg|limb|extrem)|'
                        'nog(?:a|e|u)|femoral(?:|ne) art|'
                        'do(?:|lj)nj(?:i|ih|eg) ekstrem'),
                from_val='EXTREMITY'),
            # cystography is a procedure to visualize the urinary bladder
            RegexReplacement(
                'URINARYTRACT', ('urotract|urotrakt|urinarytract|' +
                                 'urografija|cistograf|cystograph|' +
                                 'anterogradna urog|uretrografija|urogram')),
            # stenvers is a projection for observing the skull
            # tm zglob - Temporomandibular joint (mandible)
            # RegexReplacement(
            #     'HEAD', 'skull|schädel|stenvers|mandible|orbita|tm zglob'),
            RegexReplacement('CAROTID', '(?:k|c)arotid'),
            # kemoembolizacija, kolangiografija => procedures which
            # impact the liver
            # RegexReplacement(
            #     'LIVER', ('kemoemboliza|kemoembolizacija|liver|'
            #               'kolangiografija')),
            RegexReplacement(
                'URINARYTRACT', ('kemoemboliza|kemoembolizacija|liver|'
                                 'kolangiografija')),
            # KINDEY
            # renal == from kidney
            # RegexReplacement('KIDNEY', 'renal|kidney|bubre(?:g|zn)'),
            RegexReplacement('URINARYTRACT', 'renal|kidney|bubre(?:g|zn)'),
            # brachial plexus -> nerves in the spine
            RegexReplacement(
                'CSPINE', ('vratna_kra|vratna kra|c-spine|cervikalne kralj'),
                from_val='WHOLESPINE'),
            RegexReplacement(
                'CSPINE', ('vratna_kra|vratna kra|c-spine|cervikalne kralj'),
                'SPINE'),
            RegexReplacement(
                'CSPINE', ('vratna_kra|vratna kra|c-spine|cervikalne kralj'),
                'NECK'),
            RegexReplacement(
                'CSPINE', ('vratna_kra|vratna kra|c-spine|cervikalne kralj')),
            RegexReplacement('LSPINE', 'lumbaln(?:a|e)|l-spine|ls_kralj'),
            RegexReplacement('LSPINE', 'lumbaln(?:a|e)|l-spine|ls_kralj',
                             from_val='WHOLESPINE'),
            RegexReplacement('LSPINE', 'lumbaln(?:a|e)|l-spine|ls_kralj',
                             from_val='SPINE'),
            RegexReplacement('TSPINE', 'torakaln(?:a|e) kralj|t-spine'),
            RegexReplacement('TSPINE', 'torakaln(?:a|e) kralj|t-spine',
                             from_val='WHOLESPINE'),
            RegexReplacement('TSPINE', 'torakaln(?:a|e) kralj|t-spine',
                             from_val='SPINE'),
            RegexReplacement('SHOULDER', 'shoulder|rame'),
            RegexReplacement('CLAVICLE', 'kljucna kost|clavicle'),
            RegexReplacement('BREAST', 'dojka|dojke|grudna'),
            # subclavial artiery -> upper chest, below clavicles.
            # pulmonary -> related to or affecting the lungs.
            RegexReplacement(
                'CHEST', ('thorax|lung|torax|toraks|src(?:a|e)|sternum|'
                          'ribs|rebra|pluc(?:a|na)|myocardial|cardiac|'
                          'subklavij|pulmonary|pulmonal|lung|heart')),
            RegexReplacement(
                'CHEST', ('thorax|lung|torax|toraks|src(?:a|e)|sternum|'
                          'ribs|rebra|pluc(?:a|na)|myocardial|cardiac|'
                          'subklavij|pulmonary|pulmonal|lung|heart'),
                from_val='NECK'),
            RegexReplacement(
                'HIP', ('femur sa kukom|femur with hip joint'
                        '|(?:desni|lijevi|rtg|rdg) kuk')),
            # iliac artery is the main artery of the pelvis
            # si zglob - sarcal-iliac joint (pelvic area)
            RegexReplacement(
                'PELVIS', 'pelvis|zdjelic|ilijak|iliac|sacrum|si zglob'),
            RegexReplacement(
                'PELVIS', 'pelvis|zdjelic|ilijak|iliac|sacrum|si zglob',
                from_val='ABDOMEN'),
            RegexReplacement('NECK', 'neck|vrat'),
            RegexReplacement('ABDOMEN', ('abdom(?:en|inal)')),
            RegexReplacement('WHOLEBODY',
                             ('cijelo tijelo|cijelog tijela|whole body|'
                              'wholebody|total body')),
            # TEST
            # --> should be non-informative images, which were taken as
            # a result of calibrations, constancy tests etc
            RegexReplacement('TEST', 'constancy'),
        ]

        to_replace: Iterable[ValueReplacement] = [
            ValueReplacement('EXTREM', 'EXTREMITY'),
            ValueReplacement('EXTREMITIES', 'EXTREMITY'),
            ValueReplacement('UP_EXM', 'ARM'),
            ValueReplacement('LOW_EXM', 'LEG'),
            ValueReplacement('ABDOM', 'ABDOMEN'),
            ValueReplacement('C_SPNE', 'CSPINE'),
            ValueReplacement('L_SPNE', 'LSPINE'),
            ValueReplacement('T_SPNE', 'TSPINE'),
            ValueReplacement('LUMBAR SPNE', 'LSPINE'),
            ValueReplacement('Schulter', 'SHOULDER'),
            ValueReplacement('THORAX', 'CHEST'),
            ValueReplacement('HEART', 'CHEST'),
            # mr Sebastian mentioned we could map KIDNEY and LIVER
            # to URINARYTRACT
            ValueReplacement('KIDNEY', 'URINARYTRACT'),
            ValueReplacement('LIVER', 'URINARYTRACT'),
            # ValueReplacement('CEREBRALA', 'GITRACT'),
            ValueReplacement('PLUCA SJEDECI', 'CHEST'),
            # ValueReplacement('NECK', 'CSPINE'),
            # the colon and the urinary tract are parts of the abdomen
            # so it is an option to map them into the same thing
            # ValueReplacement('GITRACT', 'ABDOMEN'),
            # ValueReplacement('URINARYTRACT', 'ABDOMEN'),
        ]

        items_to_keep: typing.Set[str] = set()

        for replacement in to_replace:
            tags = self.parse_BodyPartExamined_replace(
                replacement.to_val,
                replacement.from_val,
                tags,
            )
            items_to_keep.add(replacement.to_val)

        for item in regexes:
            for other_attr in [
                'StudyDescription',
                'ProtocolName',
                'RequestedProcedureDescription',
            ]:
                if other_attr in tags.columns:
                    # if there are values which are still empty
                    # and ProtocolName tag (or another useful tag) is present
                    # then attempt to parse from it as well
                    tags = self.parse_BodyPartExamined_from_regex(
                        item.to_val,
                        regexpr=item.regexpr,
                        tags=tags,
                        from_value=item.from_val,
                        attribute_to_parse_from=other_attr
                    )
            items_to_keep.add(item.to_val)
        # a special case: all CR images of the NECK are
        # meant to be labelled as CSPINE. MR/CT and other modalities
        # can capture the neck organs so they should remain as NECK.
        # however CR always captures the bones, and cervical spine
        # is a bone in the neck.
        tags['BodyPartExaminedFixed'] = np.where(
            np.logical_and(
                tags['Modality'] == 'CR',
                tags['BodyPartExaminedFixed'] == 'NECK'
            ),
            'CSPINE',
            tags['BodyPartExaminedFixed']
        )
        tags['BodyPartExaminedFixed'] = np.where(
            np.logical_and(
                tags['Modality'] == 'CR',
                tags['BodyPartExaminedFixed'] == 'HEAD'
            ),
            'SKULL',
            tags['BodyPartExaminedFixed']
        )

        if (drop_others is True):
            tags.query(
                f'BodyPartExaminedFixed in {list(items_to_keep)}',
                inplace=True
            )
        tags = tags.drop(['BodyPartExamined'], axis=1)
        tags = tags.rename(
            columns={'BodyPartExaminedFixed': 'BodyPartExamined'})

        # change the class-attribute as well, if specified so
        self.tags = tags if inplace is True else self.tags

        return tags

# endregion

    def _switch_tags_inplace(self, switch_with: Iterable[Iterable]):
        columns = self.tags.columns
        ids = self.tags.index
        self.tags = pd.DataFrame(switch_with)
        if (len(columns) == np.shape(switch_with)[1]):
            self.tags.columns = columns
        self.tags.set_index(ids, inplace=True)

    def parse_arraylike_values_from_column(
        self,
        col_name: str,
        n_values: int,
        tags: pd.DataFrame = None,
        drop_old_col: bool = True,
        inplace: bool = False,
        verbose: bool = False,
    ):
        """If there are columns in the dataset which
        contain array-like values, then this is the function to use.
        For example, if the column contains values such as
        "['a', 'b', 'c', 'd']", then this function can transform
        this column into separate columns, where column 1
        has the value 'a', column 2 has the value 'b' and so on.

        Args:
            col_name (str): The name of the column which can have
                array-like strings.
            n_values (int): How many values should be extracted from
                the stringified array? If this is "2", then two new
                columns will be created, with the first column containing
                the first value of the array, and the second
                column containing the second value of the array.
                All other values in the array (if they exist) are ignored.
                If n_values is bigger then the actual length of the array
                (e.g., the array has 2 values but n_values is 4),
                then these excess columns will be filled with empty values.
            tags (pd.DataFrame, optional): The dataframe on which
                to perform the operation. If None, the class-attribute
                will be used. If the class-attribute is not specified,
                then an error will be thrown.
                Defaults to None.
            drop_old_col (bool, optional): After the operation is done,
                should the original column be dropped from the dataset?
                If True, it will be dropped, otherwise it will remain in
                the dataframe.
                Defaults to True.
            inplace (bool, optional): If True, the class-attribute will
                be updated as well.
                Defaults to False.
            verbose (bool, optional): Should useful logs be printed
                along the way (progress of the operation etc. will
                be printed). Defaults to False.

        Raises:
            ValueError: If there are duplicates in the dataframe.

        Returns:
            pd.DataFrame: The parsed dataframe.
        """
        tags = self._check_arg_tags(tags)

        for i in range(n_values):
            _new_col_name = f'{col_name}{i}'
            if _new_col_name not in tags.columns:
                # for example, let's say we're parsing PixelSpacing.
                # if PixelSpacing0 already exists in the dataframe
                # then nothing will be changed.
                # if PixelSpacing0 does not exist in the dataframe
                # then it will be initialized with empty strings.
                log(f'Adding column {_new_col_name}', verbose=verbose)
                tags[_new_col_name] = None

        taglen = len(tags)
        cnt_processed = 0
        log(f'Begin processing of {taglen} entries', verbose=verbose)
        for dcm_id in tags.index:
            cnt_processed += 1
            percent_completed = cnt_processed / taglen
            percent_completed *= 100
            # percent_completed = round(percent_completed, 8)

            if (
                cnt_processed % np.ceil(taglen * 0.01) == 0
            ):
                log(
                    f'Processed {percent_completed:3.3}%',
                    end='\r',
                    verbose=verbose
                )
            cur_val = tags.loc[dcm_id][col_name]

            new_val = cur_val
            try:
                new_val = re.sub("[\[\]']", '', new_val)
            except TypeError as e:
                raise ValueError(
                    f'For some reason, regex was not passed a string object.' +
                    f' The most likely scenario is that the entry' +
                    f' under ID {dcm_id} is a duplicate.' +
                    f' Please check if that is the case. Thanks!'
                )

            new_val = new_val.split(',')
            for i in range(n_values):
                if (i >= len(new_val)):
                    new_val.append('')
                    continue
                try:
                    new_val[i] = new_val[i].strip()
                except Exception as e:
                    new_val[i] = ''
            # after parsing of individual values is complete
            # then add them back to the dataframe
            for i in range(n_values):
                _new_col_name = f'{col_name}{i}'
                tags.at[dcm_id, _new_col_name] = new_val[i]

        # finally, drop the old column (if necessary)
        # because it is parsed into multiple other columns
        # and this info becomes pretty much redundant
        if (drop_old_col is True):
            log(f'Dropping column {col_name}', verbose=verbose)
            tags.drop([col_name], axis=1, inplace=True)

        if inplace is True:
            self._switch_tags_inplace(tags)
        log('All done!', verbose=verbose)
        return tags

    def parse_arraylike_and_numeric_values_from_columns(
        self,
        tags: pd.DataFrame = None,
        inplace: bool = False,
        verbose: bool = False
    ):
        """Parse multiple columns which either have
        array-like or numeric values.

        These columns are defined within the function body.
        See the implementation of this function for details.

        Args:
            tags (pd.DataFrame, optional): The dataframe to parse.
                If None, then the class-attribute will be used.
                Defaults to None.
            inplace (bool, optional): Should the class-attribute
                be replaced after this operation finishes.
                Defaults to False.
            verbose (bool, optional): If True, print useful logs.
                Defaults to False.

        Returns:
            _type_: _description_
        """
        tags = self._check_arg_tags(tags)

        numeric_el_cnt = dict()
        numeric_el_cnt['PixelSpacing'] = 2
        numeric_el_cnt['ImagerPixelSpacing'] = 2
        # these are commented out for testing purposes
        numeric_el_cnt['AcquisitionMatrix'] = 4
        numeric_el_cnt['WindowCenter'] = 1
        numeric_el_cnt['WindowWidth'] = 1
        numeric_el_cnt['ImageOrientationPatient'] = 3
        numeric_el_cnt['ImagePositionPatient'] = 3
        numeric_el_cnt['ContrastFlowDuration'] = 2
        numeric_el_cnt['ContrastFlowRate'] = 2
        numeric_el_cnt['FieldOfViewDimensions'] = 1
        numeric_el_cnt['ImageType'] = 2
        numeric_el_cnt['PatientOrientation'] = 2
        numeric_el_cnt['SequenceVariant'] = 3

        for column in tags:
            log(f'Parsing column {column}', verbose=verbose)
            if (column == 'PatientAge'):
                for dcm_id in tags.index:
                    cur_val = tags.loc[dcm_id][column]
                    new_val = cur_val
                    try:
                        # find out if we're talking about days, years or months
                        modifier = cur_val[-1]
                        # remove everything which isnt a digit
                        new_val = re.sub('[^0-9]', '', new_val)
                        new_val = float(new_val)
                        # convert all ages to 'years'
                        # see DICOM "PatientAge" format for details
                        if (modifier == 'M'):
                            # convert from Months to Years
                            new_val /= 12
                        elif (modifier == 'D'):
                            # convert from Days to Years
                            new_val /= 365
                        tags.at[dcm_id, column] = new_val
                    except Exception as e:
                        continue
            if column in numeric_el_cnt.keys():
                tags = self.parse_arraylike_values_from_column(
                    col_name=column,
                    n_values=numeric_el_cnt[column],
                    inplace=inplace,
                    drop_old_col=True,
                    verbose=verbose
                )

        if inplace is True:
            self._switch_tags_inplace(tags)
        return tags

    def drop_columns_with_insufficient_unique_values(
        self,
        tags: pd.DataFrame = None,
        inplace: bool = False,
        verbose: bool = False,
        threshold: int = 2
    ):
        """Drop columns that have too few unique values.

        Args:
            tags (pd.DataFrame, optional): The dataframe on which to
                perform the operation. Defaults to None.
            inplace (bool, optional): Should the operation be done inplace.
                Defaults to False.
            verbose (bool, optional): Whether to print useful logs.
                Defaults to False.
            threshold (int, optional): The least amount of non-empty
                unique values
                which the column should have in order to not be dropped.
                E.g., if this is 2, then any column with less than
                2 unique values will be dropped from the dataset.
                Defaults to 2.

        Returns:
            pandas.DataFrame: the transformed dataframe.
        """
        tags = self._check_arg_tags(tags)
        # remove all columns where there is only one unique value
        # (if there is only one unique value, then this column
        # holds no information)
        columns_with_insufficient_unique_values = []
        for column in tags.columns:
            unique_values_of_column = list(tags[column].unique())
            if ('' in unique_values_of_column):
                unique_values_of_column.remove('')
            if (len(unique_values_of_column) < threshold):
                columns_with_insufficient_unique_values.append(column)

        if len(columns_with_insufficient_unique_values) > 0:
            log(
                f'Dropping {len(columns_with_insufficient_unique_values)}' +
                f' columns that have less than {threshold} non-empty ' +
                f'unique values',
                verbose=verbose
            )
            tags = tags.drop(
                columns_with_insufficient_unique_values,
                axis=1
            )

        if inplace is True:
            self._switch_tags_inplace(tags)
        return tags

    def get_stats_of_columns(
        self,
        tags: pd.DataFrame = None,
        verbose: bool = False
    ):
        """Calculate different statistics of each column in dataframe.

        For now, the statistics being calculated are:
        - Number of distinct (unique) values.
            All values are calculated case insensitive.
        - Number of empty values (not available, NA values)
        - Percent how many entries have any value written. Range is 0-1
        - Entropy of column. How "messy" and unpredictable are the values
            written in each column. Calculated with a log base of 2.
        - data type of column, str or float

        Args:
            tags (pd.DataFrame, optional): The dataframe. Defaults to None.
            verbose (bool, optional): Should useful logs be printed out.
                Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe containing what was described above.
        """
        tags = self._check_arg_tags(tags)

        result = pd.DataFrame(
            columns=['UniqueValues', 'NAcount',
                     'PercentFilled', 'Entropy', 'DataType'],
            # we need to copy() the column names because
            # otherwise we will just pass a reference
            # and any changes to the index in "result" df
            # will propagate to the original reference.
            index=tags.columns.copy()
        )
        result.index.names = ['Column']
        log(
            f'Calculate distribution statistics for ' +
            f'{len(tags.columns)} columns.',
            verbose=verbose
        )
        from scipy.stats import entropy
        for column in tags:
            val_cnts = tags[column].astype('str').str.lower().value_counts()
            result.at[column, 'UniqueValues'] = len(val_cnts)
            nr_of_na_values = 0
            # calculate probabilities of all possible values
            # this will be used to calculate the entropy of the column
            class_probablities = [val_cnts.loc[val] /
                                  float(len(tags)) for val in val_cnts.index]
            result.at[column, 'Entropy'] = entropy(class_probablities, base=2)

            if ("" in val_cnts.index):
                nr_of_na_values = val_cnts[""]

            # Find out how many empty values
            result.at[column, 'NAcount'] = nr_of_na_values
            result.at[column, 'PercentFilled'] = 1 - \
                float(nr_of_na_values) / len(tags)

            # finally, check the data type
            try:
                tags[column].replace('', np.nan, regex=True).astype('float64')
                result.at[column, 'DataType'] = 'float'
            except Exception as e:
                result.at[column, 'DataType'] = 'str'

        result.sort_values(by='UniqueValues', inplace=True)

        log('Done calculating statistics!', verbose=verbose)
        return result

    def encode(
        self,
        type: str,
        tags: pd.DataFrame = None,
        ignore_empty: bool = True,
        inplace: bool = False,
        verbose: bool = False,
        return_encoder: bool = False,
    ):
        """Encode the label-like columns from dataframe
        into numeric values.

        Args:
            type (str): Which type of encoder to use. Supported values:
                `'OneHot'` for `OneHotEncoder`,
                `'Ordinal'` for `OrdinalEncoder`.
                All encoders are from existing
                implementations found in `sklearn.preprocessing`.
                When encoding, all of the categorical
                (categorial = string-like)
                features will be encoded, while numeric ones will be left
                alone.

            tags (pd.DataFrame, optional): Dicom tags which should be
                subjected to the operation. If None, then the class attribute
                `tags` will be used instead. Defaults to None.

            return_encoder (bool, optional): If `True`, then the encoder(s)
                will be returned alongside the
                operation result. Defaults to `False`.

            inplace (bool, optional): If `True`, then the class-attribute
                `self.tags` will be set to the operation result.
                Defaults to `False`.

            ignore_empty (bool, optional): If True, then empty values will
                be excluded when any encoding is performed.
                Otherwise, NaN values will be encoded as well.
                Only works with Ordinal encoding.
                Defaults to False.

            verbose (bool, optional): Whether to print
                useful logs. Defaults to False.

        Raises:
            AssertionError: If both the `tags` argument and `self.tags`
                are `None`.
            AssertionError: If the encoder type is not supported.
            e: If something happens during `encoder.fit_transform`.

        Returns:
            list: The operation result.
            encoder(s): Returned only if `return_encoder` is `True`.
                The encoders will be returned as dict,
                where each key is a column from `tags`, and value
                is the encoder for that column.
        """
        tags = self._check_arg_tags(tags)

        columns = tags.columns
        columns_to_drop = []
        encoders = dict()
        for i, column in enumerate(columns):
            try:
                tags[column] = tags[column].replace(
                    '', np.nan, regex=True)
                tags[column] = tags[column].astype('float64')
            except Exception as e:
                tags[column] = tags[column].astype('str')
                tags[column] = tags[column].str.lower()
                tags[column] = tags[column].replace(
                    'nan', '', regex=True)
                log(f'Performing {type} encoding of labels...',
                    verbose=verbose)
                if type == 'OneHot':
                    # the column is now one-hot encoded
                    encoder = prep.LabelBinarizer().fit(tags[column])
                elif type == 'Ordinal':
                    encoder = prep.LabelEncoder().fit(tags[column])
                else:
                    raise AssertionError('Invalid encoder type "%s"' % (type))
                encoders[column] = encoder
                encoded_col = encoder.transform(tags[column])
                encoded_col = np.array(encoded_col).astype('float16')
                # in the dataframe, create column
                # for each of the one-hot encoded features
                # for example, if a feature has 3 possible values
                # then encoded_col is has three columns, with
                # either zeros or ones.
                # then the resulting dataframe should get three columns
                # as well, each column containg 0 or 1
                # depending on whether this sample is part of the feature
                if type == 'OneHot':
                    column_names = []
                    for col_idx in range(encoded_col.shape[1]):
                        column_names.append(
                            f'{column}_{encoder.classes_[col_idx]}')
                    new_columns = pd.DataFrame(
                        encoded_col,
                        columns=column_names,
                        index=tags.index
                    )
                    tags = pd.concat([tags, new_columns], axis=1)
                    # drop the original column since it is replaced by
                    # three new columns
                    columns_to_drop.append(column)
                    del new_columns
                elif type == 'Ordinal':
                    tags[column] = encoded_col
                    if ignore_empty is False and '' in encoder.classes_:
                        encoding_for_empty = encoder.transform([''])[0]
                        tags[column].replace(
                            encoding_for_empty, '', inplace=True)
                del encoded_col

        if len(columns_to_drop) > 0:
            tags = tags.drop(columns=columns_to_drop)
        # change the class-attribute as well, if specified so
        if inplace is True:
            self._switch_tags_inplace(tags)
        retval = tags if return_encoder is False else (tags, encoders)

        return retval

    def scale(
        self,
        type: str,
        tags: pd.DataFrame = None,
        return_scaler: bool = False,
        inplace: bool = False
    ):
        """Scale the numeric values of a dataframe.

        Args:
            type (str): Which type of scaler to use. Supported values:
                `'MaxAbs'` for `MaxAbsScaler`,
                `'MinMax'` for `MinMaxScaler`.
                `'Standard'` for `StandardScaler`.
                `'Robust'` for `RobustScaler`.
                All scalers are from existing
                implementations found in `sklearn.preprocessing`.

            tags (pd.DataFrame, optional): Dicom tags which should be
                subjected to the operation. If None, then the class attribute
                `tags` will be used instead. Defaults to None.

            return_scaler (bool, optional): If `True`, then the scaler will
                be returned alongside the
                operation result. Defaults to `False`.

            inplace (bool, optional): If `True`, then the class-attribute
                `self.tags` will be set to the operation result.
                Also, the scaler will be instantiated without copying
                the input arguments.
                Defaults to `False`.

        Raises:
            AssertionError: If both the `tags` argument and `self.tags`
                are `None`.
            AssertionError: If the scaler type is not supported.

        Returns:
            list: The operation result.
            scaler: Returned only if `return_scaler` is `True`.
        """
        tags = self._check_arg_tags(tags)

        if type == 'MinMax':
            scaler = prep.MinMaxScaler(copy=~inplace)
        elif type == 'Robust':
            scaler = prep.RobustScaler(copy=~inplace)
        elif type == 'Standard':
            scaler = prep.StandardScaler(copy=~inplace)
        elif type == 'MaxAbs':
            scaler = prep.MaxAbsScaler(copy=~inplace)
        else:
            raise AssertionError('Invalid scaler type "%s"' % (type))

        # perform the scaling
        scaled = scaler.fit_transform(tags)
        scaled = pd.DataFrame(
            scaled,
            columns=tags.columns.copy()
        )
        scaled.set_index(tags.index.copy(), inplace=True)

        # change the class-attribute as well, if specified so
        if inplace is True:
            self._switch_tags_inplace(scaled)

        # return both the scaled values and the scaler, if specified so.
        retval = (scaled, scaler) if return_scaler is True else scaled
        return retval

    def impute_missForest(
        self,
        tags: pd.DataFrame = None,
        cat_varnames: typing.List[str] = [],
        inplace: bool = False,
        verbose: bool = False,
        random_state: int = 1,
        max_iter: int = 10,
        n_estimators: int = 256,
    ):
        """Perform imputation of missing data using MissForest.

        Args:
            tags (pd.DataFrame, optional): Data on which to perform imputation.
                Defaults to None.
            cat_varnames (typing.List[str], optional): Names of the
                categorical variables.
                In other words, this are names of columns which can be
                found in `tags`. Defaults to [].
            inplace (bool, optional): Should the operation be
                performed in place. Defaults to False.
            verbose (bool, optional): Print useful logs.
                Defaults to False.
            random_state (int, optional): For reproducibilty.
                Defaults to 1.
            max_iter (int, optional): Maximum number of iterations
                for MissForest. Defaults to 10.
            n_estimators (int, optional): Number of estimators in MissForest.
                Defaults to 256.

        Returns:
            pd.Dataframe: dataframe containing imputed data.
        """    
        tags = self._check_arg_tags(tags)
        # perform missForest imputation of missing data.
        # import the library containing MissForrest
        # to import it, you need to do some magic related to sklearn
        # otherwise miss forest complains that there is no module
        # 'sklearn.neighbors.base'
        import sklearn.neighbors._base
        import sys
        sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
        from missingpy import MissForest
        miss_forrest = MissForest(
            criterion=('squared_error', 'gini'), verbose=False,
            random_state=random_state,
            n_estimators=n_estimators,
            max_iter=max_iter
        )

        # find out the indices of categorical variables
        indices_of_categorical_vars = [
            list(tags.columns).index(cat) for cat in cat_varnames]
        indices_of_categorical_vars = None if len(
            indices_of_categorical_vars) < 1 else indices_of_categorical_vars

        log('Beginning to impute...', verbose=verbose)
        # pass the categorical var indices to fit()
        # at this point, when fit_trainsform()
        # is completed, the result is an array.
        # so after imputation is done, convert it back into a
        # dataframe
        imputed = miss_forrest.fit_transform(
            tags, cat_vars=indices_of_categorical_vars)
        # store the index (DCM IDs) for later
        index = tags.index.copy()
        tags = pd.DataFrame(
            imputed, columns=tags.columns)
        tags.set_index(index, inplace=True)
        log('Imputing done!!', verbose=verbose)

        if inplace is True:
            self._switch_tags_inplace(tags)
        return tags

    def do_pca(
        self,
        n_components: Union[float, str, None] = None,
        tags: pd.DataFrame = None,
        return_pca: bool = False,
        inplace: bool = False,
    ):
        """Perform Principal Component Analysis on the given tags.

        Args:
            n_components (Union[float, str, None], optional): How
                many components to calculate with the PCA algorithm.
                See sklearn.decomposion.PCA for details.
                Defaults to None.

            tags (pd.DataFrame, optional): Dicom tags which should be
                subjected to the operation. If None, then the class attribute
                `tags` will be used instead. Defaults to None.

            return_pca (bool, optional): If `True`, then the PCA calculator
                will be returned alongside the
                operation result. Defaults to `False`.

            inplace (bool, optional): If `True`, then the class-attribute
                `self.tags` will be set to the operation result.
                Also, the PCA calculator will be instantiated without copying
                the input arguments.
                Defaults to `False`.

        Raises:
            AssertionError: If both the `tags` argument and `self.tags`
                are `None`.

        Returns:
            list: The operation result.
            pca: Returned only if `return_pca` is `True`.
        """
        tags = self._check_arg_tags(tags)
        pca = decomp.PCA(n_components=n_components, copy=~inplace)
        transformed = pca.fit_transform(tags)

        # change the class-attribute as well, if specified so
        self.tags = transformed if inplace is True else self.tags

        retval = (transformed, pca) if return_pca is True else transformed
        return retval

    def do_ica(
        self,
        n_components: int = None,
        algorithm: str = 'parallel',
        max_iter: int = 1000,
        tags: pd.DataFrame = None,
        return_ica: bool = False,
        inplace: bool = False,
    ):
        """Perform Independent Component Analysis on the given tags.

        Args:
            n_components (int, optional): How
                many components to calculate with the FastICA algorithm.
                See sklearn.decomposion.FastICA for details.
                Defaults to None.

            algorithm (str, optional): Which algorithm to use for ICA.
                See sklearn.decomposion.FastICA for details.
                Defaults to 'parallel'.

            max_iter (int, optional): The maximum number of iterations.
                Defaults to 1000.

            tags (pd.DataFrame, optional): Dicom tags which should be
                subjected to the operation. If None, then the class attribute
                `tags` will be used instead. Defaults to None.

            return_ica (bool, optional): If `True`, then the ICA calculator
                will be returned alongside the
                operation result. Defaults to `False`.

            inplace (bool, optional): If `True`, then the class-attribute
                `self.tags` will be set to the operation result.
                Defaults to `False`.

        Raises:
            AssertionError: If both the `tags` argument and `self.tags`
                are `None`.

        Returns:
            list: The operation result.
            ica: Returned only if `return_ica` is `True`.
        """
        tags = self._check_arg_tags(tags)
        ica = decomp.FastICA(
            n_components=n_components,
            max_iter=max_iter,
            algorithm=algorithm,
        )
        transformed = ica.fit_transform(tags)

        # change the class-attribute as well, if specified so
        self.tags = transformed if inplace is True else self.tags

        retval = (transformed, ica) if return_ica is True else transformed

        return retval

    def do_tsne(
        self,
        n_components: int = None,
        perplexity: float = 30.0,
        learning_rate: float = 'auto',
        init: str = 'random',
        random_state: int = 0,
        max_iter: int = 1000,
        n_jobs: int = None,
        tags: pd.DataFrame = None,
        return_tsne: bool = False,
        inplace: bool = False,
    ):
        """Perform Stohastic Neighbour Embedding on the given tags.

        Args:
            n_components (int, optional): How
                many components to calculate with the tSNE algorithm.
                See sklearn.manifold.TSNE for details.
                Defaults to None.

            perplexity (float, optional): See sklearn.manifold.TSNE
                for a detailed explaination.
                Defaults to 30.0.

            learning_rate (float, optional): See sklearn.manifold.TSNE
                for a detailed explaination.
                Defaults to 'auto'.

            init (str, optional): See sklearn.manifold.TSNE
                for a detailed explaination.
                Defaults to 'random'.

            random_state (int, optional): See sklearn.manifold.TSNE
                for a detailed explaination.
                Defaults to 0.

            max_iter (int, optional): The maximum number of iterations.
                Defaults to 1000.

            n_jobs (int, optional): See sklearn.manifold.MDS
                for details. Defaults to None.

            tags (pd.DataFrame, optional): Dicom tags which should be
                subjected to the operation. If None, then the class attribute
                `tags` will be used instead. Defaults to None.

            return_tsne (bool, optional): If `True`, then the tSNE calculator
                will be returned alongside the
                operation result. Defaults to `False`.

            inplace (bool, optional): If `True`, then the class-attribute
                `self.tags` will be set to the operation result.
                Defaults to `False`.

        Raises:
            AssertionError: If both the `tags` argument and `self.tags`
                are `None`.

        Returns:
            list: The operation result.
            tsne: Returned only if `return_tsne` is `True`.
        """

        tags = self._check_arg_tags(tags)
        tsne = manifold.TSNE(
            n_components=n_components,
            n_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            init=init,
            n_jobs=n_jobs,
            perplexity=perplexity
        )
        transformed = tsne.fit_transform(tags)

        # change the class-attribute as well, if specified so
        self.tags = transformed if inplace is True else self.tags

        retval = (transformed, tsne) if return_tsne is True else transformed

        return retval

    def do_mds(
        self,
        n_components: int = None,
        metric: bool = True,
        n_init: int = 4,
        random_state: int = 0,
        max_iter: int = 1000,
        n_jobs: int = None,
        tags: pd.DataFrame = None,
        return_mds: bool = False,
        inplace: bool = False,
    ):
        """Perform multidimensional scaling (MDS).

        Args:
            n_components (int, optional): How
                many components to calculate with the MDS algorithm.
                See sklearn.manifold.MDS for details.
                Defaults to None.

            metric (bool, optional): See sklearn.manifold.MDS
                for details. Defaults to True.

            n_init (int, optional): See sklearn.manifold.MDS
                for details. Defaults to 4.

            random_state (int, optional): See sklearn.manifold.MDS
                for details. Defaults to 0.

            max_iter (int, optional): See sklearn.manifold.MDS
                for details. Defaults to 1000.

            n_jobs (int, optional): See sklearn.manifold.MDS
                for details. Defaults to None.

            tags (pd.DataFrame, optional): Dicom tags which should be
                subjected to the operation. If None, then the class attribute
                `tags` will be used instead. Defaults to None.

            return_mds (bool, optional): If `True`, then the MDS calculator
                will be returned alongside the
                operation result. Defaults to `False`.

            inplace (bool, optional): If `True`, then the class-attribute
                `self.tags` will be set to the operation result.
                Defaults to `False`.

        Returns:
            _type_: _description_
        """
        tags = self._check_arg_tags(tags)
        mds = manifold.MDS(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            n_init=n_init,
            metric=metric
        )
        transformed = mds.fit_transform(tags)

        # change the class-attribute as well, if specified so
        self.tags = transformed if inplace is True else self.tags

        retval = (transformed, mds) if return_mds is True else transformed

        return retval

    def resample_to_have_equal_of(
        self,
        attribute: str = 'Modality',
        tags: pd.DataFrame = None,
        nsamples: int = None,
        random_state: int = 1,
        inplace: bool = False
    ):
        """Sample the dataset so there are equal occurences
        of values of a specified attribute.

        For example, if there are 3 different unique values of
        the attribute 'Modality' and the values are [7, 8, 9].
        If the nsamples is 2000, then the resulting dataframe
        will contain 2000 occurences of rows where Modality=7,
        2000 occurences where Modality = 8 and so on.

        Args:
            attribute (str, optional): Attribute to balance.
                Defaults to 'Modality'.
            tags (pd.DataFrame, optional): Tags on which to perform
                balancing. Defaults to None.
            nsamples (int, optional): How many occurences of each
                value should be in the final result. Defaults to None.
                If this is None, then it will be calculated as the
                value which appears the least.
            random_state (int, optional): For randomness. Defaults to 1.
            inplace (bool, optional): Whether to perform
                the operation in-place. Defaults to False.

        Returns:
            The resulting dataframe.
        """
        tags = self._check_arg_tags(tags)
        unique_values = tags[attribute].unique()
        if (nsamples is None):
            nsamples = len(tags) + 1
            for value in unique_values:
                cnt = len(tags.query(f'{attribute} == "{value}"'))
                nsamples = min(cnt, nsamples)

        result = pd.DataFrame(columns=tags.columns)
        for value in unique_values:
            tmp = tags.query(f'{attribute} == "{value}"').sample(
                nsamples, random_state=random_state)
            result = pd.concat([result, tmp])

        if inplace is True:
            self._switch_tags_inplace(tags)

        return result
