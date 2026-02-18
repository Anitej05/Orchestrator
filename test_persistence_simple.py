"""
Simple Persistence Test

Tests the core file persistence functionality without running the full orchestrator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_thread_persistence():
    """Test thread workspace persistence."""
    print("\n" + "="*70)
    print("TEST 1: Thread Workspace Persistence")
    print("="*70)
    
    from backend.orchestrator.workspace_manager import get_workspace_manager
    
    # Create workspace for a thread
    thread_id = 'persistence_demo_thread'
    wm = get_workspace_manager(thread_id)
    
    print(f"Thread workspace: {wm.get_workspace_path()}")
    
    # Create a file (simulating what Python/Terminal would do)
    test_file = wm.get_workspace_path() / "analysis_results.txt"
    test_file.write_text("Q4 Sales Analysis:\n- Revenue: $1.2M\n- Growth: 15%")
    
    # Scan for new files (simulating what Hands does)
    new_files = wm.scan_for_new_files(created_by="python")
    
    print(f"Files created: {len(new_files)}")
    for f in new_files:
        print(f"  - {f.file_name} ({f.size_bytes} bytes)")
    
    # Verify file persists
    files = wm.list_files()
    print(f"Total tracked files: {len(files)}")
    
    if files:
        print("[PASS] Thread workspace persistence: WORKING")
        return True
    else:
        print("[FAIL] Files not tracked")
        return False


def test_shared_persistence():
    """Test shared workspace persistence across conversations."""
    print("\n" + "="*70)
    print("TEST 2: Shared Workspace Persistence (Cross-Conversation)")
    print("="*70)
    
    from backend.orchestrator.shared_workspace import get_shared_workspace_manager
    
    # Simulate "Conversation 1" - create and share a file
    print("\n[Conversation 1] Creating shared file...")
    swm = get_shared_workspace_manager('demo_user')
    
    # Create a file that should persist
    template_file = swm.get_workspace_path() / "email_template.txt"
    template_file.write_text("Subject: {{subject}}\n\nDear {{name}},\n\n{{message}}\n\nBest regards,")
    
    # Track it
    swm.scan_for_new_files()
    swm.add_file(
        file_path=str(template_file),
        file_name="email_template.txt",
        file_type="text/plain",
        created_by="user_shared",
        description="Reusable email template"
    )
    
    files_convo1 = swm.list_files()
    print(f"Files in shared workspace: {[f.file_name for f in files_convo1]}")
    
    # Simulate "Conversation 2" - new thread, same user
    print("\n[Conversation 2] Accessing shared file from new conversation...")
    
    # Create new workspace manager (simulating new conversation)
    swm2 = get_shared_workspace_manager('demo_user')
    
    files_convo2 = swm2.list_files()
    print(f"Files accessible: {[f.file_name for f in files_convo2]}")
    
    # Verify file is accessible
    if files_convo2 and any('email_template' in f.file_name for f in files_convo2):
        print("[PASS] Shared workspace persistence: WORKING")
        print("  - Files created in Conversation 1")
        print("  - Accessible in Conversation 2")
        print("  - Cross-conversation persistence: CONFIRMED")
        return True
    else:
        print("[FAIL] Shared file not accessible")
        return False


def test_user_isolation():
    """Test that users are isolated from each other."""
    print("\n" + "="*70)
    print("TEST 3: User Isolation")
    print("="*70)
    
    from backend.orchestrator.shared_workspace import get_shared_workspace_manager
    
    # User 1 creates a file
    swm_user1 = get_shared_workspace_manager('user_alice')
    secret_file = swm_user1.get_workspace_path() / "alice_private.txt"
    secret_file.write_text("Alice's private data")
    swm_user1.scan_for_new_files()
    
    # User 2 tries to access
    swm_user2 = get_shared_workspace_manager('user_bob')
    bob_files = swm_user2.list_files()
    
    print(f"Alice's files: {[f.file_name for f in swm_user1.list_files()]}")
    print(f"Bob's files: {[f.file_name for f in bob_files]}")
    
    if not any('alice_private' in f.file_name for f in bob_files):
        print("[PASS] User isolation: WORKING")
        print("  - Alice's files are private")
        print("  - Bob cannot access them")
        return True
    else:
        print("[FAIL] User isolation broken")
        return False


def test_workspace_structure():
    """Show the complete workspace structure."""
    print("\n" + "="*70)
    print("WORKSPACE STRUCTURE OVERVIEW")
    print("="*70)
    
    from backend.orchestrator.workspace_manager import ORCHESTRATOR_WORKSPACE
    from backend.orchestrator.shared_workspace import SHARED_WORKSPACE
    
    print("\n📁 STORAGE STRUCTURE:")
    print(f"\n  backend/storage/")
    print(f"  ├── orchestrator/          ← Thread workspaces (private)")
    print(f"  │   ├── thread_001/        ← Conversation 1 files")
    print(f"  │   ├── thread_002/        ← Conversation 2 files")
    print(f"  │   └── ...")
    print(f"  ├── shared/                ← Cross-conversation workspaces")
    print(f"  │   ├── user_alice/        ← Alice's persistent files")
    print(f"  │   ├── user_bob/          ← Bob's persistent files")
    print(f"  │   └── ...")
    print(f"  ├── browser_agent/         ← Agent workspaces")
    print(f"  ├── spreadsheet_agent/")
    print(f"  └── ...")
    
    print(f"\n📍 ACTUAL PATHS:")
    print(f"  Thread workspace base: {ORCHESTRATOR_WORKSPACE}")
    print(f"  Shared workspace base: {SHARED_WORKSPACE}")
    
    # List actual contents
    if ORCHESTRATOR_WORKSPACE.exists():
        threads = [d.name for d in ORCHESTRATOR_WORKSPACE.iterdir() if d.is_dir()]
        print(f"\n  Active thread workspaces: {threads}")
    
    if SHARED_WORKSPACE.exists():
        users = [d.name for d in SHARED_WORKSPACE.iterdir() if d.is_dir()]
        print(f"  Active shared workspaces: {users}")
    
    return True


def main():
    """Run all persistence tests."""
    print("\n" + "="*70)
    print("FILE PERSISTENCE SYSTEM - FUNCTIONAL TESTS")
    print("="*70)
    print("Testing thread persistence, shared persistence, and isolation")
    
    results = []
    
    # Test 1: Thread persistence
    results.append(("Thread Persistence", test_thread_persistence()))
    
    # Test 2: Shared persistence
    results.append(("Shared Persistence", test_shared_persistence()))
    
    # Test 3: User isolation
    results.append(("User Isolation", test_user_isolation()))
    
    # Show structure
    test_workspace_structure()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe persistence system is working correctly:")
        print("  1. ✅ Thread workspace - Files persist within conversation")
        print("  2. ✅ Shared workspace - Files persist across conversations")
        print("  3. ✅ User isolation - Users cannot see each other's files")
        print("\nKey Features:")
        print("  • Temporary by default (thread workspace)")
        print("  • Persistent when needed (shared workspace)")
        print("  • Automatic file tracking")
        print("  • Secure user isolation")
    else:
        print(f"\n⚠️ {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
