CREATE EXTENSION pg_stat_statements;
CREATE ROLE router LOGIN PASSWORD 'load-test';
GRANT CREATE ON SCHEMA public TO router;
