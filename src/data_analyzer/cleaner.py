# src/data_analyzer/cleaner.py

import pandas as pd
import logging  # <--- 新增导入
from typing import Optional, List  # <--- 新增 List 类型

# 配置 logger（实际项目中通常在入口文件配置，这里简单演示）
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class used to clean tabular data using pandas.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize with a pandas DataFrame.

        :param df: Input pandas DataFrame
        """
        self.df = df

    def load_csv(self, filepath: str) -> None:
        """
        Load data from a CSV file.

        :param filepath: Path to the CSV file
        """
        try:
            self.df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")

    def clean_data(
        self, drop_na_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:  # noqa: E501
        """
        :param drop_na_cols: List of column names to check for NaNs.
                            If None, drops rows where ANY column is NaN.

        Perform standard cleaning operations:
        1. Normalize column names (lowercase, spaces to underscores)
        2. Remove duplicate rows
        3. Drop rows with missing values

        :return: Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")

        # 1. Normalize column names
        self.df.columns = self.df.columns.str.lower().str.strip()

        # 2. Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        dedup_rows = len(self.df)

        # 3. Drop missing values (更灵活的逻辑)
        if drop_na_cols:
            self.df = self.df.dropna(subset=drop_na_cols)
        else:
            self.df = self.df.dropna()

        final_rows = len(self.df)
        # 替换 print 为 logging
        logger.info(
            f"Cleaned data: Removed {initial_rows - dedup_rows} duplicates, "
            f"{dedup_rows - final_rows} rows with missing values."
        )

        return self.df
