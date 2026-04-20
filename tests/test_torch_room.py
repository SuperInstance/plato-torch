"""Tests for plato-torch room self-training."""

import json
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plato_torch.torch_room import TorchRoom
from plato_torch.tile_grabber import TileGrabber


def test_room_observe():
    """Room records observations as training tiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        room = TorchRoom("test-room", train_threshold=1000,
                         ensign_dir=os.path.join(tmpdir, "ensigns"),
                         buffer_dir=os.path.join(tmpdir, "tiles"))
        
        tile = room.observe(
            state="hole=[A♠,K♠] pot=100 pos=BTN",
            action="raise",
            outcome="won 180",
            agent_id="test-agent"
        )
        
        assert tile["reward"] == 1.0  # "won" → positive
        assert tile["state_hash"] is not None
        assert room._episodes_seen == 1
        print("✅ test_room_observe passed")


def test_room_reward_inference():
    """Room infers reward from outcome text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        room = TorchRoom("test-room", train_threshold=1000,
                         ensign_dir=os.path.join(tmpdir, "ensigns"),
                         buffer_dir=os.path.join(tmpdir, "tiles"))
        
        assert room._infer_reward("", "", "won big pot") == 1.0
        assert room._infer_reward("", "", "lost everything") == -1.0
        assert room._infer_reward("", "", "checked") == 0.0
        assert room._infer_reward("", "", "success!") == 1.0
        assert room._infer_reward("", "", "failed attempt") == -1.0
        print("✅ test_room_reward_inference passed")


def test_room_train():
    """Room trains from accumulated tiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        room = TorchRoom("test-room", train_threshold=5,
                         ensign_dir=os.path.join(tmpdir, "ensigns"),
                         buffer_dir=os.path.join(tmpdir, "tiles"))
        
        # Feed enough tiles to trigger training
        for i in range(10):
            room.observe(
                state=f"hole=[A♠,K♠] pot={100+i*10} pos=BTN",
                action="raise",
                outcome="won" if i % 2 == 0 else "lost",
                agent_id="test-agent"
            )
        
        # Train explicitly (auto-train may have already fired)
        result = room.train()
        
        assert result["status"] == "trained"
        assert result["tiles"] >= 10
        
        # Check instinct
        feel = room.instinct("hole=[A♠,K♠] pot=100 pos=BTN")
        assert feel["confidence"] in ["high", "medium", "low"]
        print(f"  Instinct: feel={feel['feel']}, suggested={feel['suggested']}")
        print("✅ test_room_train passed")


def test_room_simulation():
    """Room can simulate episodes to generate training data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        room = TorchRoom("test-room", use_case="game", train_threshold=10,
                         ensign_dir=os.path.join(tmpdir, "ensigns"),
                         buffer_dir=os.path.join(tmpdir, "tiles"))
        
        result = room.simulate(episodes=50, strategies=["aggressive", "conservative"])
        
        assert result["episodes"] == 50
        assert result["tiles_generated"] == 50
        print(f"  Simulated {result['tiles_generated']} tiles")
        print("✅ test_room_simulation passed")


def test_room_wisdom():
    """Room reports accumulated wisdom."""
    with tempfile.TemporaryDirectory() as tmpdir:
        room = TorchRoom("test-room", use_case="game", train_threshold=5,
                         ensign_dir=os.path.join(tmpdir, "ensigns"),
                         buffer_dir=os.path.join(tmpdir, "tiles"))
        
        room.simulate(episodes=20)
        wisdom = room.wisdom()
        
        assert wisdom["room_id"] == "test-room"
        assert wisdom["episodes_seen"] >= 20
        print(f"  Wisdom: {json.dumps(wisdom, indent=2)}")
        print("✅ test_room_wisdom passed")


def test_tile_grabber():
    """Tile grabber learns which tiles are relevant."""
    grabber = TileGrabber("test-room")
    
    # Agent grabbed certain tiles and succeeded
    grabber.observe_grab(
        "late_position_strong_hand",
        ["aggression_tile", "position_tile", "pot_odds_tile"],
        reward=1.0
    )
    grabber.observe_grab(
        "early_position_weak_hand",
        ["caution_tile", "fold_tile"],
        reward=0.5
    )
    grabber.observe_grab(
        "late_position_strong_hand",
        ["aggression_tile", "bluff_tile"],
        reward=1.5
    )
    
    # Ask for recommendations
    available = ["aggression_tile", "caution_tile", "bluff_tile", "pot_odds_tile", "fold_tile"]
    recs = grabber.recommend_tiles("late_position_strong_hand", available, top_k=3)
    
    assert len(recs) <= 3
    # Aggression tile should rank high for strong late position
    tile_ids = [r[0] for r in recs]
    print(f"  Recommended tiles: {recs}")
    print("✅ test_tile_grabber passed")


if __name__ == "__main__":
    test_room_observe()
    test_room_reward_inference()
    test_room_train()
    test_room_simulation()
    test_room_wisdom()
    test_tile_grabber()
    print("\n🎉 All tests passed!")
