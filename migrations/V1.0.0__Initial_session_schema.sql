-- Initial session management schema
-- Version: 1.0.0
-- Description: Initial session management schema with chat sessions table

-- Create chat_sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT NOT NULL,
    reasoning_mode TEXT NOT NULL,
    messages TEXT NOT NULL,
    metadata TEXT,
    is_archived BOOLEAN DEFAULT FALSE,
    tags TEXT,
    user_id TEXT DEFAULT 'default'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON chat_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_model ON chat_sessions(model_used);
CREATE INDEX IF NOT EXISTS idx_sessions_reasoning_mode ON chat_sessions(reasoning_mode);
CREATE INDEX IF NOT EXISTS idx_sessions_archived ON chat_sessions(is_archived);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON chat_sessions(updated_at);

-- Add comments for documentation
PRAGMA table_info(chat_sessions); 