import folktables
from folktables import ACSDataSource

from aif360.datasets import StandardDataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

from .dataloader import get_dataloaders


def get_distortion_acs_income(vold, vnew):
    """Distortion function for the ACS Income Folktables dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    def adjustAge(a):
        return float(a)

    def adjustwkhp(a):
        return int(a)

    def adjustLabel(a):
        if a == True:
            return 1.0
        else:
            return 0.0
    
    # value that will be returned for events that should not occur
    bad_val = 3.0

    # adjust age
    aOld = adjustAge(vold['AGEP'])
    aNew = adjustAge(vnew['AGEP'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld-aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld-aNew) > 1e-3:
        return 2.0


    # Adjust hours worked per week
    wkhpOld = adjustwkhp(vold['WKHP'])
    wkhpNew = adjustwkhp(vnew['WKHP'])

    if np.abs(wkhpOld-wkhpNew) > 10.0:
        return bad_val

    labelOld = adjustLabel(vold['PINCP'])
    labelNew = adjustLabel(vnew['PINCP'])

    if labelOld > labelNew:
        return 1.0
    else:
        return 0.0


def get_acsincome_dataloaders(train_bs=64, test_bs=64, privilege_mode='SEX', scale=True):
    privileged_groups, unprivileged_groups = [{privilege_mode: 1}], [{privilege_mode: 0}]
    
    dataset_orig = ACSIncomeFolktablesDataset(protected_attr_name=privilege_mode)
    data_orig_train, data_orig_val = dataset_orig.split([0.65], shuffle=True)
    data_orig_val, data_orig_test = data_orig_val.split([0.43], shuffle=True) # 0.43 * 0.35 = 0.15

    if scale:
        preproc = MaxAbsScaler()
        data_orig_train.features = preproc.fit_transform(data_orig_train.features)
        data_orig_val.features = preproc.fit_transform(data_orig_val.features)
        data_orig_test.features = preproc.fit_transform(data_orig_test.features)

    train_loader, val_loader, test_loader = get_dataloaders(data_orig_train, data_orig_val,
                                                            data_orig_test, train_bs=train_bs,
                                                            test_bs=test_bs)
    data = {
        'task_type': 'bin-class',
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': data_orig_train,
        'val_dataset': data_orig_val,
        'test_dataset': data_orig_test,
        'privileged_groups': privileged_groups,
        'unprivileged_groups': unprivileged_groups
    }
    return data


class ACSIncomeFolktablesDataset(StandardDataset):
    def __init__(self, protected_attr_name='SEX', root_dir="data/folktables/",
                 survey_year='2018', states=["PA"]):
        assert protected_attr_name in ['SEX', 'RAC1P']
        self.protected_attr_name = protected_attr_name

        # 'sex': 'SEX',
        # 'race': 'RAC1P'

        def group_race(x):
            if x == 1.:
                return 1.
            else:
                return 0.

        data_source = ACSDataSource(survey_year=survey_year, horizon='1-Year', survey='person', root_dir=root_dir)
        acs_data = data_source.get_data(states=states, download=True)
        features, labels, _ = self.formulate_problem().df_to_pandas(acs_data)

        df = pd.concat([features, labels], axis=1)
        df['SEX'] = df['SEX'].replace({1.: 1., 2.: 0.})
        df['RAC1P'] = df['RAC1P'].apply(lambda x: group_race(x))

        del features
        del labels

        super(ACSIncomeFolktablesDataset, self).__init__(df=df, label_name='PINCP', favorable_classes=[True], 
                                            protected_attribute_names=[protected_attr_name],
                                            privileged_classes=[[1.]], 
                                            categorical_features=['COW', 'MAR', 'SCHL'])
    
    def formulate_problem(self):
        problem = folktables.BasicProblem(
            features=[
                'AGEP',
                'COW',    # class of worker
                'SCHL',   # education level
                'MAR',    # marital status
                # 'OCCP', # occupation
                # 'POBP', # place of birth
                # 'RELP', # relationship 
                'WKHP', 
                'SEX',
                'RAC1P',
            ],
            target='PINCP',
            target_transform=lambda x: x > 50000,
            group=self.protected_attr_name,
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return problem
