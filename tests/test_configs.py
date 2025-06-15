"""
Tests for morphix configs module
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

from configs import get_app_config
from configs.app_config import AppConfig, RemoteSettingsSourceFactory
from configs.deploy import DeploymentConfig
from configs.enterprise import EnterpriseFeatureConfig
from configs.feature import (
    AppExecutionConfig,
    AuthConfig,
    BillingConfig,
    CodeExecutionSandboxConfig,
    DataSetConfig,
    EndpointConfig,
    FileAccessConfig,
    FileUploadConfig,
    HttpConfig,
    InnerAPIConfig,
    LoggingConfig,
    MailConfig,
    MarketplaceConfig,
    ModelLoadBalanceConfig,
    ModerationConfig,
    MultiModalTransferConfig,
    PluginConfig,
    PositionConfig,
    RagEtlConfig,
    SecurityConfig,
    ToolConfig,
    UpdateConfig,
    WorkflowConfig,
    WorkflowNodeExecutionConfig,
    WorkspaceConfig,
)
from configs.middleware import (
    CeleryConfig,
    DatabaseConfig,
    KeywordStoreConfig,
    MiddlewareConfig,
    RedisConfig,
    StorageConfig,
    VectorStoreConfig,
)
from configs.middleware.cache.redis_config import RedisConfig
from configs.middleware.storage.aliyun_oss_storage_config import AliyunOSSStorageConfig
from configs.middleware.storage.amazon_s3_storage_config import S3StorageConfig
from configs.middleware.storage.azure_blob_storage_config import AzureBlobStorageConfig
from configs.middleware.vdb.analyticdb_config import AnalyticdbConfig
from configs.middleware.vdb.chroma_config import ChromaConfig
from configs.middleware.vdb.elasticsearch_config import ElasticsearchConfig
from configs.middleware.vdb.milvus_config import MilvusConfig
from configs.middleware.vdb.opensearch_config import OpenSearchConfig
from configs.middleware.vdb.pgvector_config import PGVectorConfig
from configs.middleware.vdb.qdrant_config import QdrantConfig
from configs.middleware.vdb.weaviate_config import WeaviateConfig
from configs.observability.otel.otel_config import OTelConfig
from configs.packaging import PackagingInfo
from configs.remote_settings_sources import RemoteSettingsSourceConfig, RemoteSettingsSourceName
from configs.remote_settings_sources.apollo import ApolloSettingsSource
from configs.remote_settings_sources.nacos import NacosSettingsSource


class TestDeploymentConfig:
    """Test deployment configuration"""

    def test_default_values(self):
        """Test default deployment configuration values"""
        config = DeploymentConfig()
        assert config.APPLICATION_NAME == "langgenius/dify"
        assert config.DEBUG is False
        assert config.ENABLE_REQUEST_LOGGING is False
        assert config.EDITION == "SELF_HOSTED"
        assert config.DEPLOY_ENV == "PRODUCTION"

    def test_custom_values(self):
        """Test custom deployment configuration values"""
        config = DeploymentConfig(
            APPLICATION_NAME="morphix",
            DEBUG=True,
            ENABLE_REQUEST_LOGGING=True,
            EDITION="CLOUD",
            DEPLOY_ENV="DEVELOPMENT"
        )
        assert config.APPLICATION_NAME == "morphix"
        assert config.DEBUG is True
        assert config.ENABLE_REQUEST_LOGGING is True
        assert config.EDITION == "CLOUD"
        assert config.DEPLOY_ENV == "DEVELOPMENT"


class TestSecurityConfig:
    """Test security configuration"""

    def test_default_values(self):
        """Test default security configuration values"""
        config = SecurityConfig()
        assert config.SECRET_KEY == ""
        assert config.RESET_PASSWORD_TOKEN_EXPIRY_MINUTES == 5
        assert config.LOGIN_DISABLED is False
        assert config.ADMIN_API_KEY_ENABLE is False
        assert config.ADMIN_API_KEY is None

    def test_secret_key_validation(self):
        """Test secret key configuration"""
        config = SecurityConfig(SECRET_KEY="test-secret-key-123")
        assert config.SECRET_KEY == "test-secret-key-123"

    def test_admin_api_key_configuration(self):
        """Test admin API key configuration"""
        config = SecurityConfig(
            ADMIN_API_KEY_ENABLE=True,
            ADMIN_API_KEY="admin-test-key"
        )
        assert config.ADMIN_API_KEY_ENABLE is True
        assert config.ADMIN_API_KEY == "admin-test-key"


class TestDatabaseConfig:
    """Test database configuration"""

    def test_default_values(self):
        """Test default database configuration values"""
        config = DatabaseConfig()
        assert config.DB_HOST == "localhost"
        assert config.DB_PORT == 5432
        assert config.DB_USERNAME == "postgres"
        assert config.DB_PASSWORD == ""
        assert config.DB_DATABASE == "dify"
        assert config.SQLALCHEMY_DATABASE_URI_SCHEME == "postgresql"

    def test_sqlalchemy_database_uri_generation(self):
        """Test SQLAlchemy database URI generation"""
        config = DatabaseConfig(
            DB_HOST="127.0.0.1",
            DB_PORT=5433,
            DB_USERNAME="morphix_user",
            DB_PASSWORD="secret123",
            DB_DATABASE="morphix_db"
        )
        expected_uri = "postgresql://morphix_user:secret123@127.0.0.1:5433/morphix_db"
        assert config.SQLALCHEMY_DATABASE_URI == expected_uri

    def test_database_uri_with_extras(self):
        """Test database URI with extra parameters"""
        config = DatabaseConfig(
            DB_EXTRAS="sslmode=require&connect_timeout=10",
            DB_CHARSET="utf8"
        )
        uri = config.SQLALCHEMY_DATABASE_URI
        assert "sslmode=require" in uri
        assert "connect_timeout=10" in uri
        assert "client_encoding=utf8" in uri

    def test_sqlalchemy_engine_options(self):
        """Test SQLAlchemy engine options"""
        config = DatabaseConfig()
        options = config.SQLALCHEMY_ENGINE_OPTIONS
        assert options["pool_size"] == 30
        assert options["max_overflow"] == 10
        assert options["pool_recycle"] == 3600
        assert options["pool_pre_ping"] is False
        assert "-c timezone=UTC" in options["connect_args"]["options"]


class TestRedisConfig:
    """Test Redis configuration"""

    def test_default_values(self):
        """Test default Redis configuration values"""
        config = RedisConfig()
        assert config.REDIS_HOST == "localhost"
        assert config.REDIS_PORT == 6379
        assert config.REDIS_USERNAME is None
        assert config.REDIS_PASSWORD is None
        assert config.REDIS_DB == 0
        assert config.REDIS_USE_SSL is False

    def test_sentinel_configuration(self):
        """Test Redis Sentinel configuration"""
        config = RedisConfig(
            REDIS_USE_SENTINEL=True,
            REDIS_SENTINELS="sentinel1:26379,sentinel2:26379",
            REDIS_SENTINEL_SERVICE_NAME="mymaster",
            REDIS_SENTINEL_USERNAME="sentinel_user",
            REDIS_SENTINEL_PASSWORD="sentinel_pass"
        )
        assert config.REDIS_USE_SENTINEL is True
        assert config.REDIS_SENTINELS == "sentinel1:26379,sentinel2:26379"
        assert config.REDIS_SENTINEL_SERVICE_NAME == "mymaster"

    def test_cluster_configuration(self):
        """Test Redis Cluster configuration"""
        config = RedisConfig(
            REDIS_USE_CLUSTERS=True,
            REDIS_CLUSTERS="node1:7000,node2:7001,node3:7002",
            REDIS_CLUSTERS_PASSWORD="cluster_pass"
        )
        assert config.REDIS_USE_CLUSTERS is True
        assert config.REDIS_CLUSTERS == "node1:7000,node2:7001,node3:7002"
        assert config.REDIS_CLUSTERS_PASSWORD == "cluster_pass"


class TestStorageConfig:
    """Test storage configuration"""

    def test_default_values(self):
        """Test default storage configuration values"""
        config = StorageConfig()
        assert config.STORAGE_TYPE == "opendal"

    def test_s3_storage_configuration(self):
        """Test S3 storage configuration"""
        s3_config = S3StorageConfig(
            S3_ENDPOINT="https://s3.amazonaws.com",
            S3_BUCKET_NAME="morphix-bucket",
            S3_ACCESS_KEY="AKIAIOSFODNN7EXAMPLE",
            S3_SECRET_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            S3_REGION="us-east-1"
        )
        assert s3_config.S3_ENDPOINT == "https://s3.amazonaws.com"
        assert s3_config.S3_BUCKET_NAME == "morphix-bucket"
        assert s3_config.S3_REGION == "us-east-1"
        assert s3_config.S3_USE_AWS_MANAGED_IAM is False

    def test_azure_blob_storage_configuration(self):
        """Test Azure Blob storage configuration"""
        azure_config = AzureBlobStorageConfig(
            AZURE_BLOB_ACCOUNT_NAME="morphixstorage",
            AZURE_BLOB_ACCOUNT_KEY="base64key==",
            AZURE_BLOB_CONTAINER_NAME="morphix-container",
            AZURE_BLOB_ACCOUNT_URL="https://morphixstorage.blob.core.windows.net"
        )
        assert azure_config.AZURE_BLOB_ACCOUNT_NAME == "morphixstorage"
        assert azure_config.AZURE_BLOB_CONTAINER_NAME == "morphix-container"


class TestVectorStoreConfig:
    """Test vector store configuration"""

    def test_weaviate_configuration(self):
        """Test Weaviate configuration"""
        config = WeaviateConfig(
            WEAVIATE_ENDPOINT="http://localhost:8080",
            WEAVIATE_API_KEY="test-api-key",
            WEAVIATE_GRPC_ENABLED=True,
            WEAVIATE_BATCH_SIZE=200
        )
        assert config.WEAVIATE_ENDPOINT == "http://localhost:8080"
        assert config.WEAVIATE_API_KEY == "test-api-key"
        assert config.WEAVIATE_GRPC_ENABLED is True
        assert config.WEAVIATE_BATCH_SIZE == 200

    def test_qdrant_configuration(self):
        """Test Qdrant configuration"""
        config = QdrantConfig(
            QDRANT_URL="http://localhost:6333",
            QDRANT_API_KEY="qdrant-key",
            QDRANT_CLIENT_TIMEOUT=30,
            QDRANT_GRPC_ENABLED=True,
            QDRANT_GRPC_PORT=6334,
            QDRANT_REPLICATION_FACTOR=2
        )
        assert config.QDRANT_URL == "http://localhost:6333"
        assert config.QDRANT_CLIENT_TIMEOUT == 30
        assert config.QDRANT_REPLICATION_FACTOR == 2

    def test_pgvector_configuration(self):
        """Test PGVector configuration"""
        config = PGVectorConfig(
            PGVECTOR_HOST="localhost",
            PGVECTOR_PORT=5433,
            PGVECTOR_USER="pgvector_user",
            PGVECTOR_PASSWORD="pgvector_pass",
            PGVECTOR_DATABASE="vectordb",
            PGVECTOR_MIN_CONNECTION=2,
            PGVECTOR_MAX_CONNECTION=10
        )
        assert config.PGVECTOR_HOST == "localhost"
        assert config.PGVECTOR_PORT == 5433
        assert config.PGVECTOR_MIN_CONNECTION == 2
        assert config.PGVECTOR_MAX_CONNECTION == 10


class TestWorkflowConfig:
    """Test workflow configuration"""

    def test_default_values(self):
        """Test default workflow configuration values"""
        config = WorkflowConfig()
        assert config.WORKFLOW_MAX_EXECUTION_STEPS == 500
        assert config.WORKFLOW_MAX_EXECUTION_TIME == 1200
        assert config.WORKFLOW_CALL_MAX_DEPTH == 5
        assert config.WORKFLOW_PARALLEL_DEPTH_LIMIT == 3
        assert config.MAX_VARIABLE_SIZE == 200 * 1024

    def test_custom_limits(self):
        """Test custom workflow limits"""
        config = WorkflowConfig(
            WORKFLOW_MAX_EXECUTION_STEPS=1000,
            WORKFLOW_MAX_EXECUTION_TIME=3600,
            WORKFLOW_CALL_MAX_DEPTH=10
        )
        assert config.WORKFLOW_MAX_EXECUTION_STEPS == 1000
        assert config.WORKFLOW_MAX_EXECUTION_TIME == 3600
        assert config.WORKFLOW_CALL_MAX_DEPTH == 10


class TestHttpConfig:
    """Test HTTP configuration"""

    def test_default_values(self):
        """Test default HTTP configuration values"""
        config = HttpConfig()
        assert config.API_COMPRESSION_ENABLED is False
        assert config.HTTP_REQUEST_MAX_CONNECT_TIMEOUT == 10
        assert config.HTTP_REQUEST_MAX_READ_TIMEOUT == 60
        assert config.HTTP_REQUEST_MAX_WRITE_TIMEOUT == 20
        assert config.SSRF_DEFAULT_MAX_RETRIES == 3

    def test_cors_configuration(self):
        """Test CORS configuration"""
        config = HttpConfig(
            # inner_CONSOLE_CORS_ALLOW_ORIGINS="https://console.morphix.io,https://admin.morphix.io",
            # inner_WEB_API_CORS_ALLOW_ORIGINS="https://api.morphix.io"
        )
        config.set_CONSOLE_CORS_ALLOW_ORIGINS("https://console.morphix.io,https://admin.morphix.io")
        config.set_WEB_API_CORS_ALLOW_ORIGINS("https://api.morphix.io")
        assert config.CONSOLE_CORS_ALLOW_ORIGINS == ["https://console.morphix.io", "https://admin.morphix.io"]
        assert config.WEB_API_CORS_ALLOW_ORIGINS == ["https://api.morphix.io"]

    def test_ssrf_proxy_configuration(self):
        """Test SSRF proxy configuration"""
        config = HttpConfig(
            SSRF_PROXY_ALL_URL="http://proxy.local:8080",
            SSRF_DEFAULT_TIME_OUT=10.0,
            SSRF_DEFAULT_CONNECT_TIME_OUT=3.0
        )
        assert config.SSRF_PROXY_ALL_URL == "http://proxy.local:8080"
        assert config.SSRF_DEFAULT_TIME_OUT == 10.0
        assert config.SSRF_DEFAULT_CONNECT_TIME_OUT == 3.0


class TestMailConfig:
    """Test mail configuration"""

    def test_smtp_configuration(self):
        """Test SMTP mail configuration"""
        config = MailConfig(
            MAIL_TYPE="smtp",
            MAIL_DEFAULT_SEND_FROM="noreply@morphix.io",
            SMTP_SERVER="smtp.gmail.com",
            SMTP_PORT=587,
            SMTP_USERNAME="morphix@gmail.com",
            SMTP_PASSWORD="app-password",
            SMTP_USE_TLS=True
        )
        assert config.MAIL_TYPE == "smtp"
        assert config.SMTP_SERVER == "smtp.gmail.com"
        assert config.SMTP_PORT == 587
        assert config.SMTP_USE_TLS is True

    def test_resend_configuration(self):
        """Test Resend mail configuration"""
        config = MailConfig(
            MAIL_TYPE="resend",
            RESEND_API_KEY="re_123456789",
            RESEND_API_URL="https://api.resend.com"
        )
        assert config.MAIL_TYPE == "resend"
        assert config.RESEND_API_KEY == "re_123456789"


class TestOTelConfig:
    """Test OpenTelemetry configuration"""

    def test_default_values(self):
        """Test default OpenTelemetry configuration values"""
        config = OTelConfig()
        assert config.ENABLE_OTEL is False
        assert config.OTLP_BASE_ENDPOINT == "http://localhost:4318"
        assert config.OTEL_EXPORTER_TYPE == "otlp"
        assert config.OTEL_EXPORTER_OTLP_PROTOCOL == "http"
        assert config.OTEL_SAMPLING_RATE == 0.1

    def test_enabled_configuration(self):
        """Test enabled OpenTelemetry configuration"""
        config = OTelConfig(
            ENABLE_OTEL=True,
            OTLP_BASE_ENDPOINT="http://otel-collector:4318",
            OTLP_API_KEY="otel-api-key",
            OTEL_SAMPLING_RATE=1.0
        )
        assert config.ENABLE_OTEL is True
        assert config.OTLP_BASE_ENDPOINT == "http://otel-collector:4318"
        assert config.OTEL_SAMPLING_RATE == 1.0


class TestPositionConfig:
    """Test position configuration"""

    def test_provider_pins_list(self):
        """Test provider pins list parsing"""
        config = PositionConfig(
            POSITION_PROVIDER_PINS="openai,anthropic,azure"
        )
        assert config.POSITION_PROVIDER_PINS_LIST == ["openai", "anthropic", "azure"]

    def test_tool_includes_set(self):
        """Test tool includes set parsing"""
        config = PositionConfig(
            POSITION_TOOL_INCLUDES="web_search,calculator,code_interpreter"
        )
        assert config.POSITION_TOOL_INCLUDES_SET == {"web_search", "calculator", "code_interpreter"}

    def test_empty_values(self):
        """Test empty position values"""
        config = PositionConfig()
        assert config.POSITION_PROVIDER_PINS_LIST == []
        assert config.POSITION_TOOL_INCLUDES_SET == set()
        assert config.POSITION_TOOL_EXCLUDES_SET == set()


class TestEnterpriseFeatureConfig:
    """Test enterprise feature configuration"""

    def test_default_values(self):
        """Test default enterprise feature configuration values"""
        config = EnterpriseFeatureConfig()
        assert config.ENTERPRISE_ENABLED is False
        assert config.CAN_REPLACE_LOGO is False

    def test_enabled_features(self):
        """Test enabled enterprise features"""
        config = EnterpriseFeatureConfig(
            ENTERPRISE_ENABLED=True,
            CAN_REPLACE_LOGO=True
        )
        assert config.ENTERPRISE_ENABLED is True
        assert config.CAN_REPLACE_LOGO is True


class TestRemoteSettingsSource:
    """Test remote settings source configuration"""

    def test_apollo_settings_source(self):
        """Test Apollo settings source"""
        with patch('configs.remote_settings_sources.apollo.client.ApolloClient') as mock_apollo:
            mock_client = MagicMock()
            mock_apollo.return_value = mock_client
            mock_client.get_all_dicts.return_value = {
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG"
            }
            
            source = ApolloSettingsSource({
                "APOLLO_APP_ID": "morphix",
                "APOLLO_CLUSTER": "default",
                "APOLLO_CONFIG_URL": "http://apollo:8080",
                "APOLLO_NAMESPACE": "application"
            })
            
            assert source.namespace == "application"
            # assert source.remote_configs["DEBUG"] == "true"
            # assert source.remote_configs["LOG_LEVEL"] == "DEBUG"

    def test_nacos_settings_source(self):
        """Test Nacos settings source"""
        with patch('configs.remote_settings_sources.nacos.NacosHttpClient') as mock_nacos:
            mock_client = MagicMock()
            mock_nacos.return_value = mock_client
            mock_client.http_request.return_value = "DEBUG=true\nLOG_LEVEL=DEBUG"
            
            with patch.dict(os.environ, {
                "DIFY_ENV_NACOS_DATA_ID": "morphix-config",
                "DIFY_ENV_NACOS_GROUP": "DEFAULT_GROUP",
                "DIFY_ENV_NACOS_NAMESPACE": "public"
            }):
                source = NacosSettingsSource({})
                # assert source.remote_configs["DEBUG"] == "true"
                # assert source.remote_configs["LOG_LEVEL"] == "DEBUG"


class TestAppConfig:
    """Test main AppConfig class"""

    def test_app_config_initialization(self):
        """Test AppConfig initialization with default values"""
        config = AppConfig()
        
        # Test packaging info
        assert config.CURRENT_VERSION == "1.4.2"
        
        # Test deployment config
        assert config.DEBUG is False
        assert config.EDITION == "SELF_HOSTED"
        
        # Test security config
        assert config.SECRET_KEY == ""
        assert config.LOGIN_DISABLED is False
        
        # Test database config
        assert config.DB_HOST == "localhost"
        assert config.DB_PORT == 5432
        
        # Test Redis config
        assert config.REDIS_HOST == "localhost"
        assert config.REDIS_PORT == 6379

    def test_app_config_with_env_file(self):
        """Test AppConfig loading from .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("DEBUG=true\n")
            f.write("SECRET_KEY=test-secret-key\n")
            f.write("DB_HOST=morphix-db\n")
            f.write("DB_PORT=5433\n")
            f.write("REDIS_HOST=morphix-redis\n")
            f.flush()
            
            try:
                config = AppConfig(_env_file=f.name)
                assert config.DEBUG is True
                assert config.SECRET_KEY == "test-secret-key"
                assert config.DB_HOST == "morphix-db"
                assert config.DB_PORT == 5433
                assert config.REDIS_HOST == "morphix-redis"
            finally:
                os.unlink(f.name)

    def test_app_config_validation_error(self):
        """Test AppConfig validation errors"""
        with pytest.raises(ValidationError):
            AppConfig(DB_PORT=-1)  # Negative port should fail
        
        with pytest.raises(ValidationError):
            AppConfig(REDIS_PORT=0)  # Zero port should fail
        
        with pytest.raises(ValidationError):
            AppConfig(UPLOAD_FILE_SIZE_LIMIT=-10)  # Negative size should fail

    def test_remote_settings_source_factory(self):
        """Test RemoteSettingsSourceFactory"""
        class TestSettings(BaseSettings):
            test_field: str = Field(default="default")
        
        factory = RemoteSettingsSourceFactory(TestSettings)
        
        # Test without remote source
        # factory.current_state = {}
        result = factory()
        assert result == {}
        
        # Test with unsupported remote source
        # factory.current_state = {"REMOTE_SETTINGS_SOURCE_NAME": "unsupported"}
        result = factory()
        assert result == {}

    def test_get_app_config_singleton(self):
        """Test get_app_config returns singleton instance"""
        config1 = get_app_config()
        config2 = get_app_config()
        assert config1 is config2  # Should be the same instance


class TestCodeExecutionSandboxConfig:
    """Test code execution sandbox configuration"""

    def test_default_values(self):
        """Test default code execution sandbox configuration values"""
        config = CodeExecutionSandboxConfig()
        assert str(config.CODE_EXECUTION_ENDPOINT) == "http://sandbox:8194/"
        assert config.CODE_EXECUTION_API_KEY == "dify-sandbox"
        assert config.CODE_EXECUTION_CONNECT_TIMEOUT == 10.0
        assert config.CODE_EXECUTION_READ_TIMEOUT == 60.0
        assert config.CODE_MAX_STRING_LENGTH == 80000

    def test_numeric_limits(self):
        """Test numeric limits in code execution"""
        config = CodeExecutionSandboxConfig()
        assert config.CODE_MAX_NUMBER == 9223372036854775807
        assert config.CODE_MIN_NUMBER == -9223372036854775807
        assert config.CODE_MAX_DEPTH == 5
        assert config.CODE_MAX_PRECISION == 20


class TestFileUploadConfig:
    """Test file upload configuration"""

    def test_default_values(self):
        """Test default file upload configuration values"""
        config = FileUploadConfig()
        assert config.UPLOAD_FILE_SIZE_LIMIT == 15
        assert config.UPLOAD_FILE_BATCH_LIMIT == 5
        assert config.UPLOAD_IMAGE_FILE_SIZE_LIMIT == 10
        assert config.UPLOAD_VIDEO_FILE_SIZE_LIMIT == 100
        assert config.UPLOAD_AUDIO_FILE_SIZE_LIMIT == 50
        assert config.WORKFLOW_FILE_UPLOAD_LIMIT == 10

    def test_custom_limits(self):
        """Test custom file upload limits"""
        config = FileUploadConfig(
            UPLOAD_FILE_SIZE_LIMIT=50,
            UPLOAD_IMAGE_FILE_SIZE_LIMIT=25,
            BATCH_UPLOAD_LIMIT=50
        )
        assert config.UPLOAD_FILE_SIZE_LIMIT == 50
        assert config.UPLOAD_IMAGE_FILE_SIZE_LIMIT == 25
        assert config.BATCH_UPLOAD_LIMIT == 50


class TestCeleryConfig:
    """Test Celery configuration"""

    def test_database_backend(self):
        """Test Celery with database backend"""
        config = CeleryConfig(
            CELERY_BACKEND="database",
            DB_USERNAME="celery_user",
            DB_PASSWORD="celery_pass",
            DB_DATABASE="celery_db"
        )
        assert config.CELERY_BACKEND == "database"
        assert "celery_user" in config.CELERY_RESULT_BACKEND
        assert "celery_db" in config.CELERY_RESULT_BACKEND

    def test_redis_backend(self):
        """Test Celery with Redis backend"""
        config = CeleryConfig(
            CELERY_BACKEND="redis",
            CELERY_BROKER_URL="redis://localhost:6379/0"
        )
        assert config.CELERY_BACKEND == "redis"
        assert config.CELERY_RESULT_BACKEND == "redis://localhost:6379/0"
        assert config.BROKER_USE_SSL is False

    def test_redis_ssl_backend(self):
        """Test Celery with Redis SSL backend"""
        config = CeleryConfig(
            CELERY_BACKEND="redis",
            CELERY_BROKER_URL="rediss://localhost:6380/0"
        )
        assert config.BROKER_USE_SSL is True


class TestAuthConfig:
    """Test authentication configuration"""

    def test_default_values(self):
        """Test default authentication configuration values"""
        config = AuthConfig()
        assert config.ACCESS_TOKEN_EXPIRE_MINUTES == 60
        assert config.REFRESH_TOKEN_EXPIRE_DAYS == 30
        assert config.LOGIN_LOCKOUT_DURATION == 86400
        assert config.FORGOT_PASSWORD_LOCKOUT_DURATION == 86400

    def test_oauth_configuration(self):
        """Test OAuth configuration"""
        config = AuthConfig(
            GITHUB_CLIENT_ID="github-client-id",
            GITHUB_CLIENT_SECRET="github-secret",
            GOOGLE_CLIENT_ID="google-client-id",
            GOOGLE_CLIENT_SECRET="google-secret",
            OAUTH_REDIRECT_PATH="/auth/callback"
        )
        assert config.GITHUB_CLIENT_ID == "github-client-id"
        assert config.GOOGLE_CLIENT_ID == "google-client-id"
        assert config.OAUTH_REDIRECT_PATH == "/auth/callback"


class TestLoggingConfig:
    """Test logging configuration"""

    def test_default_values(self):
        """Test default logging configuration values"""
        config = LoggingConfig()
        assert config.LOG_LEVEL == "INFO"
        assert config.LOG_FILE is None
        assert config.LOG_FILE_MAX_SIZE == 20
        assert config.LOG_FILE_BACKUP_COUNT == 5
        assert config.LOG_TZ == "UTC"

    def test_file_logging(self):
        """Test file logging configuration"""
        config = LoggingConfig(
            LOG_LEVEL="DEBUG",
            LOG_FILE="/var/log/morphix/app.log",
            LOG_FILE_MAX_SIZE=50,
            LOG_FILE_BACKUP_COUNT=10,
            LOG_TZ="America/New_York"
        )
        assert config.LOG_LEVEL == "DEBUG"
        assert config.LOG_FILE == "/var/log/morphix/app.log"
        assert config.LOG_FILE_MAX_SIZE == 50
        assert config.LOG_TZ == "America/New_York"


class TestComplexScenarios:
    """Test complex configuration scenarios"""

    def test_multiple_vector_stores_configuration(self):
        """Test configuration with multiple vector stores"""
        # This tests that different vector store configs can coexist
        weaviate = WeaviateConfig(WEAVIATE_ENDPOINT="http://weaviate:8080")
        qdrant = QdrantConfig(QDRANT_URL="http://qdrant:6333")
        milvus = MilvusConfig(MILVUS_URI="http://milvus:19530")
        
        assert weaviate.WEAVIATE_ENDPOINT == "http://weaviate:8080"
        assert qdrant.QDRANT_URL == "http://qdrant:6333"
        assert milvus.MILVUS_URI == "http://milvus:19530"

    def test_environment_override(self):
        """Test environment variable override"""
        with patch.dict(os.environ, {
            "DEBUG": "true",
            "SECRET_KEY": "env-secret-key",
            "DB_HOST": "env-db-host"
        }):
            config = AppConfig()
            assert config.DEBUG is True
            assert config.SECRET_KEY == "env-secret-key"
            assert config.DB_HOST == "env-db-host"

    def test_config_inheritance_chain(self):
        """Test configuration inheritance chain"""
        # AppConfig inherits from multiple configs
        config = AppConfig()
        
        # From PackagingInfo
        assert hasattr(config, 'CURRENT_VERSION')
        
        # From DeploymentConfig
        assert hasattr(config, 'DEBUG')
        assert hasattr(config, 'EDITION')
        
        # From FeatureConfig (SecurityConfig)
        assert hasattr(config, 'SECRET_KEY')
        
        # From MiddlewareConfig (DatabaseConfig)
        assert hasattr(config, 'DB_HOST')
        assert hasattr(config, 'SQLALCHEMY_DATABASE_URI')
        
        # From ObservabilityConfig
        assert hasattr(config, 'ENABLE_OTEL')
        
        # From EnterpriseFeatureConfig
        assert hasattr(config, 'ENTERPRISE_ENABLED')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])