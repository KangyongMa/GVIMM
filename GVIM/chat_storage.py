import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class ChatSessionStorage:
    def __init__(self, storage_dir: str = "chat_history"):
        """Initialize the chat session storage."""
        self.storage_dir = storage_dir
        self.chat_sessions = {}
        self._ensure_storage_dir()
        self._load_sessions()

    def _ensure_storage_dir(self) -> None:
        """Ensure that the storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_sessions(self) -> None:
        """Load all existing chat sessions from storage."""
        try:
            if not os.path.exists(self.storage_dir):
                return
            
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json extension
                    file_path = os.path.join(self.storage_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        self.chat_sessions[session_id] = json.load(file)
        except Exception as e:
            print(f"Error loading chat sessions: {e}")

    def _save_session(self, session_id: str) -> bool:
        """Save the specified chat session to storage."""
        try:
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(self.chat_sessions.get(session_id, []), file, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving chat session {session_id}: {e}")
            return False

    def add_session_entry(self, session_id: str, entry: Dict[str, Any]) -> bool:
        """Add a new entry to the specified chat session."""
        try:
            if session_id not in self.chat_sessions:
                self.chat_sessions[session_id] = []
            
            self.chat_sessions[session_id].append(entry)
            return self._save_session(session_id)
        except Exception as e:
            print(f"Error adding session entry: {e}")
            return False

    def get_session_history(self, session_id: str = "all") -> Dict[str, Any]:
        """Get the history for a specific session or all sessions."""
        if session_id == "all":
            return {
                "sessions": [
                    {
                        "id": sid,
                        "entries": entries,
                        "created_at": entries[0]["timestamp"] if entries else datetime.now().isoformat(),
                        "updated_at": entries[-1]["timestamp"] if entries else datetime.now().isoformat(),
                        "entry_count": len(entries)
                    }
                    for sid, entries in self.chat_sessions.items()
                ]
            }
        else:
            entries = self.chat_sessions.get(session_id, [])
            return {
                "session": {
                    "id": session_id,
                    "entries": entries,
                    "created_at": entries[0]["timestamp"] if entries else datetime.now().isoformat(),
                    "updated_at": entries[-1]["timestamp"] if entries else datetime.now().isoformat(),
                    "entry_count": len(entries)
                }
            }

    def add_feedback(self, session_id: str, message_index: int, feedback_data: Dict[str, Any]) -> bool:
        """Add feedback to a specific message in a chat session."""
        try:
            if session_id not in self.chat_sessions:
                print(f"Session {session_id} not found")
                return False
            
            if message_index == -1:  # Add feedback to the latest entry
                if not self.chat_sessions[session_id]:
                    print(f"No entries in session {session_id}")
                    return False
                
                entry = self.chat_sessions[session_id][-1]
                
                if "feedback" not in entry:
                    entry["feedback"] = {}
                
                # Update the feedback data with the new feedback
                entry["feedback"].update(feedback_data)
                
                # Add timestamp to the feedback
                entry["feedback"]["timestamp"] = datetime.now().isoformat()
                
                return self._save_session(session_id)
            
            elif 0 <= message_index < len(self.chat_sessions[session_id]):
                entry = self.chat_sessions[session_id][message_index]
                
                if "feedback" not in entry:
                    entry["feedback"] = {}
                
                # Update the feedback data with the new feedback
                entry["feedback"].update(feedback_data)
                
                # Add timestamp to the feedback
                entry["feedback"]["timestamp"] = datetime.now().isoformat()
                
                return self._save_session(session_id)
            else:
                print(f"Message index {message_index} is out of range for session {session_id}")
                return False
        except Exception as e:
            print(f"Error adding feedback: {e}")
            return False
    
    def store_feedback(self, feedback_data: Dict[str, Any], session_id: str = "default") -> bool:
        """Store feedback for the most recent chat entry."""
        try:
            if session_id not in self.chat_sessions or not self.chat_sessions[session_id]:
                print(f"No entries found for session {session_id}")
                return False
            
            # Get the most recent entry
            entry = self.chat_sessions[session_id][-1]
            
            # Add or update the feedback field
            if "feedback" not in entry:
                entry["feedback"] = {}
            
            entry["feedback"].update(feedback_data)
            entry["feedback"]["timestamp"] = datetime.now().isoformat()
            
            return self._save_session(session_id)
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return False
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all entries for a specific session."""
        try:
            if session_id in self.chat_sessions:
                self.chat_sessions[session_id] = []
                return self._save_session(session_id)
            return True  # Session doesn't exist, so it's effectively cleared
        except Exception as e:
            print(f"Error clearing session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session completely."""
        try:
            if session_id in self.chat_sessions:
                del self.chat_sessions[session_id]
                
                # Also delete the file
                file_path = os.path.join(self.storage_dir, f"{session_id}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return True
            return False  # Session doesn't exist
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False