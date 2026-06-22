-- config.internal.lua — compatibility alias for the live policy-router catalog.
--
-- The source of truth is config.live.lua plus the three policy files under
-- policies/{edge,medium,dummy}.lua. Keep this alias so older deployment/docs
-- references do not drift into a second, contradictory catalog.

local HERE = os.getenv("LLM_POLICY_DIR") or "."
return dofile(HERE .. "/config.live.lua")
