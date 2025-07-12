"""
Data source factory for handling talent, company, and news data.
"""
import json
import pandas as pd
from typing import Dict, Any, List
from factories.base import DataSourceFactory, DataLoader
from models.talent import TalentData
import os


class TalentDataLoader(DataLoader):
    """Loader for talent JSON data."""
    
    def __init__(self, data_dir: str = "example_datas"):
        self.data_dir = data_dir
    
    def load(self, identifier: str) -> Dict[str, Any]:
        """Load talent data by identifier (e.g., 'talent_ex1')."""
        try:
            file_path = os.path.join(self.data_dir, f"{identifier}.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise ValueError(f"Talent data not found: {identifier}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in talent data {identifier}: {str(e)}")
    
    def load_all_talents(self) -> List[Dict[str, Any]]:
        """Load all talent data files."""
        talent_files = [f for f in os.listdir(self.data_dir) if f.startswith('talent_') and f.endswith('.json')]
        talents = []
        
        for file in talent_files:
            identifier = file.replace('.json', '')
            try:
                talent_data = self.load(identifier)
                talent_data['_file_id'] = identifier
                talents.append(talent_data)
            except ValueError as e:
                print(f"Warning: Could not load {identifier}: {e}")
                
        return talents


class CompanyDataLoader(DataLoader):
    """Loader for company JSON data."""
    
    def __init__(self, data_dir: str = "example_datas"):
        self.data_dir = data_dir
    
    def load(self, identifier: str) -> Dict[str, Any]:
        """Load company data by identifier (e.g., 'company_ex1_비바리퍼블리카')."""
        try:
            file_path = os.path.join(self.data_dir, f"{identifier}.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise ValueError(f"Company data not found: {identifier}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in company data {identifier}: {str(e)}")
    
    def load_by_company_name(self, company_name: str) -> Dict[str, Any]:
        """Load company data by company name."""
        # Map company names to file identifiers
        company_mapping = {
            "비바리퍼블리카": "company_ex1_비바리퍼블리카",
            "토스": "company_ex1_비바리퍼블리카",
            "네이버": "company_ex2_네이버",
            "리디": "company_ex3_리디",
            "엘박스": "company_ex4_엘박스",
            "카사코리아": "company_ex5_카사코리아",
            "야놀자": "company_ex6_야놀자",
            "시어스랩": "compnay_ex7_시어스랩"
        }
        
        identifier = company_mapping.get(company_name)
        if not identifier:
            raise ValueError(f"No data available for company: {company_name}")
        
        return self.load(identifier)


class NewsDataLoader(DataLoader):
    """Loader for news CSV data."""
    
    def __init__(self, data_dir: str = "example_datas"):
        self.data_dir = data_dir
        self.news_file = os.path.join(data_dir, "company_news.csv")
    
    def load(self, identifier: str) -> Dict[str, Any]:
        """Load news data by company identifier."""
        try:
            df = pd.read_csv(self.news_file)
            # Filter news by company name
            company_news = df[df['company_name'] == identifier]
            
            return {
                'company_name': identifier,
                'news_count': len(company_news),
                'news_data': company_news.to_dict('records')
            }
        except FileNotFoundError:
            raise ValueError(f"News data file not found: {self.news_file}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"News data file is empty: {self.news_file}")
    
    def load_all_news(self) -> Dict[str, Any]:
        """Load all news data."""
        try:
            df = pd.read_csv(self.news_file)
            return {
                'total_news_count': len(df),
                'companies': df['company_name'].unique().tolist(),
                'news_data': df.to_dict('records')
            }
        except FileNotFoundError:
            raise ValueError(f"News data file not found: {self.news_file}")


class DefaultDataSourceFactory(DataSourceFactory):
    """Default implementation of data source factory."""
    
    SUPPORTED_SOURCES = ["talent", "company", "news"]
    
    def __init__(self, data_dir: str = "example_datas"):
        self.data_dir = data_dir
        self.loaders = {
            "talent": TalentDataLoader(data_dir),
            "company": CompanyDataLoader(data_dir),
            "news": NewsDataLoader(data_dir)
        }
    
    def create_data_loader(self, source_type: str) -> DataLoader:
        """Create a data loader for the specified source type."""
        if source_type not in self.SUPPORTED_SOURCES:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return self.loaders[source_type]
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported data source types."""
        return self.SUPPORTED_SOURCES.copy()


class DataSourceManager:
    """Manager for data sources with caching."""
    
    def __init__(self, factory: DataSourceFactory):
        self.factory = factory
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def get_talent_data(self, talent_id: str, use_cache: bool = True) -> TalentData:
        """Get talent data as TalentData model."""
        cache_key = f"talent_{talent_id}"
        
        if use_cache and cache_key in self.cache:
            raw_data = self.cache[cache_key]
        else:
            loader = self.factory.create_data_loader("talent")
            raw_data = loader.load(talent_id)
            if use_cache:
                self.cache[cache_key] = raw_data
        
        return TalentData(**raw_data)
    
    def get_company_data(self, company_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get company data by name."""
        cache_key = f"company_{company_name}"
        
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        loader = self.factory.create_data_loader("company")
        data = loader.load_by_company_name(company_name)
        
        if use_cache:
            self.cache[cache_key] = data
        
        return data
    
    def get_news_data(self, company_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get news data by company name."""
        cache_key = f"news_{company_name}"
        
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        loader = self.factory.create_data_loader("news")
        data = loader.load(company_name)
        
        if use_cache:
            self.cache[cache_key] = data
        
        return data
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear() 