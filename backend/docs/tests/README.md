# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_morphix_configs.py -v

# 运行特定测试类
pytest tests/test_morphix_configs.py::TestDatabaseConfig -v

# 查看测试覆盖率
pytest tests/ --cov=configs --cov-report=html