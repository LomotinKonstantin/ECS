import os
import pandas as pd


class CodesHelper:
    def __init__(self, ipv_name=None, ipv_change=None, clear_math=True):
        self.ipv_codes = self.load_ipv_codes(ipv_name, clear_math)
        self.ipv_change = self.load_ipv_change(ipv_change)
        if clear_math:
            # print('!!! Math')
            self.subj_codes = ['e1', 'e2', 'e3', 'e4', 'e5', 'e7', 'e9',
                               'f1', 'f2', 'f3', 'f4', 'f5', 'f7', 'f8', 'f9']
        else:
            self.subj_codes = ['e1', 'e2', 'e3', 'e4', 'e5', 'e7', 'e8', 'e9',
                               'f1', 'f2', 'f3', 'f4', 'f5', 'f7', 'f8', 'f9']
    
    def clear_null(self, data, column_name):
        data.index.name = 'index'
        data = data.reset_index()
        data = data.drop(data[data[column_name].isnull()].index)
        data = data.set_index(data['index']).drop('index', axis=1)
        return data

    @staticmethod
    def load_ipv_codes(ipv_name: str, clear_math: bool):
        """
        Loads ipv codes if path is valid.

        Args:
        ipv_name    -- absolute or relative path to the ipv codes file.
        """
        if ipv_name and os.path.exists(ipv_name):
            ipv_codes = list(pd.read_csv(ipv_name, sep='\t', header=None)[0])
            if clear_math:
                math = []
                for i in ipv_codes:
                    if i.startswith('13'):
                        math += [i]
                ipv_codes = list(set(ipv_codes) - set(math))
        else:
            ipv_codes = None
        return ipv_codes

    @staticmethod
    def load_ipv_change(ipv_change_file: str):
        """
        Loads ipv changes file if path is valid.

        Args:
        ipv_change: absolute or relative path to the ipv changes file with two columns
                    separated with tab: original ipv code and it's change.
        """
        if ipv_change_file and os.path.exists(ipv_change_file):
            ipv_change = pd.read_csv(ipv_change_file, sep='\t', header=None)
        else:
            ipv_change = None
        return ipv_change
        
        #############################
#         ToDo Test
    
    def change_ipv(self, data):
        """
        Changes ipv column in pd.dataFrame according to ipv_change.

        Args:
        data          -- pd.DataFrame with ipv column.
        clear_math    -- boolean parameter. If True math SRSTI (13) will be cleared.
        """
        data = self.clear_null(data, 'ipv')
        for i in self.ipv_change.index:
            temp = list(self.ipv_change.loc[i])
            data.ipv[data.ipv == temp[0]] = temp[1]
        codes = list(set(self.ipv_codes))
        # if self.clear_math:
        #     math = []
        #     for i in self.ipv_codes:
        #         if i.startswith('13'):
        #             math += [i]
        #     codes = list(set(self.ipv_codes)-set(math))
        clear = list(set(list(data.ipv.unique()))-set(codes))
        if clear:
            for i in list(set(list(data.ipv.unique()))-set(codes)):
                idx = data[data.ipv == i].index
                if idx.size == 0:
                    data = data.drop(idx, axis=0)
        return data
    
    def change_subj(self, data):
        data = self.clear_null(data, 'subj')
        if data.subj.isnull().any():
            data.index.name = 'index'
            data = data.reset_index()
        codes = list(set(data.subj.unique())-set(self.subj_codes))
        for i in codes:
            data = data.drop(data[data.subj == i].index, axis=0)
        if data.subj.isnull().any():
            data = data.set_index(data['index']).drop('index', axis=1)
        return data
    
    def change_rgnti(self, data):
        data = self.clear_null(data, 'rgnti')
        data.rgnti = self.cut_rgnti(data.rgnti)
        return data
    #############################
    
    def get_codes(self, name):
        """
        Gives list with all valid codes of a rubricator.

        Args:
        name        -- string rubricator name in VINITI format ("SUBJ", "IPV", etc.)
        """
        if name.lower() == 'ipv':
            return self.ipv_codes
        elif name.lower() == 'subj':
            s = self.subj_codes
        else:
            print('Name must be SUBJ or IPV')
            return None
        return s

    def cut_rgnti(self, data):
        """
        Transforms rgnti cide into xx.xx format.

        Args:
        data        -- list/pd.Series with rgnti column.
        """
        for i in data.unique():
            if type(i) == str:
                data[data == str(i)] = str(i)[:5]
        return data

#
# if __name__ == '__main__':
#     ch = Codes_helper()
#     print(ch.ipv_codes)
#     print(ch.ipv_change)
