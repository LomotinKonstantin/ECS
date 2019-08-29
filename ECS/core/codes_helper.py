import os
from collections import OrderedDict

import pandas as pd


class RubricManager:
    def __init__(self,
                 ipv_subj_match_file=None,
                 ipv_update_file=None,
                 remove_math=True):
        if ipv_subj_match_file is None:
            ipv_subj_match_file = os.path.join(os.path.dirname(__file__), "RJ_code_21017_utf8.txt")
        if ipv_update_file is None:
            ipv_update_file = os.path.join(os.path.dirname(__file__), "Replacement_RJ_code_utf8.txt")
        self.ipv_subj_descr = self._load_ipv_codes(ipv_path=ipv_subj_match_file,
                                                   ipv_update_file=ipv_update_file,
                                                   remove_math=remove_math)
        self.subj = ['e1', 'e2', 'e3', 'e4', 'e5', 'e7', 'e8', 'e9',
                     'f1', 'f2', 'f3', 'f4', 'f5', 'f7', 'f8', 'f9']
        if remove_math:
            self.subj.remove("e8")

    @staticmethod
    def _load_ipv_codes(ipv_path: str,
                        ipv_update_file: str,
                        remove_math: bool) -> pd.DataFrame:
        """
        Загрузить и обновить коды IPV
        :param ipv_path:
        :param ipv_update_file:
        :param remove_math:
        :return:
        """
        ipv_match_df = pd.read_csv(ipv_path, sep='\t', names=["ipv", "subj", "descr"])
        replacement_scheme = pd.read_csv(ipv_update_file, sep='\t', names=["new"], index_col=0).to_dict()["new"]
        ipv_match_df.replace(replacement_scheme, inplace=True)
        if remove_math:
            math_indices = ipv_match_df.index[list(map(lambda x: x.startswith("13"),
                                                       ipv_match_df["ipv"]))]
            ipv_match_df.drop(index=math_indices, inplace=True)
        return ipv_match_df

    @staticmethod
    def cut_rgnti(data: pd.DataFrame) -> None:
        """
        Переводит столбец rgnti в формат xx.xx
        :param data: датафрейм со столбцом "rgnti"
        :return: None
        """
        data["rgnti"] = data["rgnti"].apply(lambda x: x[:5])

    @property
    def ipv(self):
        ipv_list = self.ipv_subj_descr["ipv"].values
        # После замены образовались дублирующиеся коды
        return list(OrderedDict.fromkeys(ipv_list))

    # @property
    # def grnti(self):
    #     return self.ipv_subj_descr["rgnti"].values
#
# if __name__ == '__main__':
#     ch = Codes_helper()
#     print(ch.ipv_codes)
#     print(ch.ipv_change)
