# app/cache_manager.py

from app.tools.qualitative_tool import QualitativeAnalysisTool
from app.tools.financial_extractor import extract_financial_metrics
from app.logger import logger
from app.tools.screener_pipeline import ScreenerPipeline
from app.tools.screener_pipeline import ScreenerPipeline
class CacheManager:
    qualitative_cache = None
    financial_cache = None

    @classmethod
    def warm_up(cls):
        try:
            logger.info("Running Screener ingestion pipeline...")
            pipeline_out = ScreenerPipeline.run()
            logger.info(f"Pipeline result: {pipeline_out}")

        except Exception as e:
            logger.error(f"Screener pipeline failed: {e}")
        try:
            logger.info("Warming up cache: financial extractor...")
            cls.financial_cache = extract_financial_metrics(None)
        except Exception as e:
            logger.error(f"Financial cache failed: {e}")

        try:
            logger.info("Warming up cache: qualitative analysis...")
            tool = QualitativeAnalysisTool()
            cls.qualitative_cache = tool.analyze()
        except Exception as e:
            logger.error(f"Qualitative cache failed: {e}")

        logger.info("Cache warmup complete.")

    @classmethod
    def get_financial(cls):
        return cls.financial_cache

    @classmethod
    def get_qualitative(cls):
        return cls.qualitative_cache
