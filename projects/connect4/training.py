import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json

from experiments import Experiment, Transition
from games import RandomStrat, LearnedPolicy, Game
from utils import MODELS_FOLDER


class TransitionDataset:
    def __init__(self, samples: List[Transition]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "boards": torch.tensor(s.boards, dtype=torch.float32),
            "action": torch.tensor(s.action, dtype=torch.long),
            "dc_return": torch.tensor(s.dc_return, dtype=torch.float32),
        }


class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x


class Trainer:
    def __init__(
        self, model: Connect4Net, lr: float = 1e-3, device: str = "mps"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history = {"loss": [], "epoch": []}
        self.max_grad_norm = 1.0

    def _check_model_health(self, prefix: str = ""):
        """Check for NaN/Inf in model parameters and track weight statistics."""
        stats = {}
        has_nan = False
        has_inf = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"  [ERROR] {prefix} NaN in parameter: {name}")
                has_nan = True
            if torch.isinf(param).any():
                print(f"  [ERROR] {prefix} Inf in parameter: {name}")
                has_inf = True
            stats[name] = {
                "mean": param.data.abs().mean().item(),
                "max": param.data.abs().max().item(),
            }
        return has_nan, has_inf, stats

    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Track metrics across the epoch
        epoch_stats = {
            "max_logit": float("-inf"),
            "min_logit": float("inf"),
            "max_grad_norm": 0.0,
            "nan_batches": 0,
            "inf_batches": 0,
            "min_entropy": float("inf"),
        }

        for batch in dataloader:
            boards = batch["boards"].to(self.device)
            actions = batch["action"].to(self.device)
            returns = batch["dc_return"].to(self.device)

            # Diagnostics: raw returns before normalization
            if n_batches == 0:
                print(
                    f"  [Diag] Raw returns - mean: {returns.mean().item():.4f}, "
                    f"std: {returns.std().item():.4f}, "
                    f"range: [{returns.min().item():.4f}, {returns.max().item():.4f}]"
                )

            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            self.optimizer.zero_grad()
            logits = self.model(boards)

            # Track logit extremes across all batches
            epoch_stats["max_logit"] = max(
                epoch_stats["max_logit"], logits.max().item()
            )
            epoch_stats["min_logit"] = min(
                epoch_stats["min_logit"], logits.min().item()
            )

            # Diagnostics: check logits for NaN/Inf
            if torch.isnan(logits).any():
                epoch_stats["nan_batches"] += 1
                if epoch_stats["nan_batches"] == 1:
                    print(
                        f"  [WARN] NaN detected in logits at batch {n_batches}!"
                    )
            if torch.isinf(logits).any():
                epoch_stats["inf_batches"] += 1
                if epoch_stats["inf_batches"] == 1:
                    print(
                        f"  [WARN] Inf detected in logits at batch {n_batches}!"
                    )

            if n_batches == 0:
                print(
                    f"  [Diag] Logits - mean: {logits.mean().item():.4f}, "
                    f"std: {logits.std().item():.4f}, "
                    f"range: [{logits.min().item():.4f}, {logits.max().item():.4f}]"
                )

            # REINFORCE: -log(π(a|s)) * R
            log_probs = torch.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            # Compute entropy for all batches to track collapse
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            epoch_stats["min_entropy"] = min(
                epoch_stats["min_entropy"], entropy.item()
            )

            # Diagnostics: log probs and policy entropy
            if n_batches == 0:
                print(
                    f"  [Diag] Action log_probs - mean: {action_log_probs.mean().item():.4f}, "
                    f"range: [{action_log_probs.min().item():.4f}, {action_log_probs.max().item():.4f}]"
                )
                print(
                    f"  [Diag] Normalized returns - mean: {returns.mean().item():.4f}, "
                    f"std: {returns.std().item():.4f}"
                )
                print(f"  [Diag] Policy entropy: {entropy.item():.4f}")
                # Check for policy collapse (entropy near 0 means deterministic)
                if entropy.item() < 0.1:
                    print(
                        "  [WARN] Very low entropy - policy may be collapsing!"
                    )

            loss = -(action_log_probs * returns).mean()

            # Diagnostics: loss components
            if n_batches == 0:
                loss_components = action_log_probs * returns
                print(
                    f"  [Diag] Loss components (log_p * R) - mean: {loss_components.mean().item():.4f}, "
                    f"range: [{loss_components.min().item():.4f}, {loss_components.max().item():.4f}]"
                )
                print(f"  [Diag] Loss value: {loss.item():.4f}")

            # Check for NaN loss before backward
            if torch.isnan(loss):
                print(f"  [ERROR] NaN loss at batch {n_batches}!")
                continue

            loss.backward()

            # Compute gradient norm before clipping
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5
            epoch_stats["max_grad_norm"] = max(
                epoch_stats["max_grad_norm"], total_norm
            )

            # Diagnostics: gradient norms before clipping
            if n_batches == 0:
                print(f"  [Diag] Gradient norm (pre-clip): {total_norm:.4f}")
            elif total_norm > 10.0:  # Alert on large gradients
                print(
                    f"  [WARN] Large gradient norm at batch {n_batches}: {total_norm:.4f}"
                )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # Diagnostics: gradient norms after clipping
            if n_batches == 0:
                total_norm_post = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm_post += p.grad.data.norm(2).item() ** 2
                total_norm_post = total_norm_post**0.5
                print(
                    f"  [Diag] Gradient norm (post-clip): {total_norm_post:.4f}"
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Epoch summary diagnostics
        avg_loss = total_loss / n_batches
        print(
            f"  [Epoch Summary] Logit range: [{epoch_stats['min_logit']:.2f}, {epoch_stats['max_logit']:.2f}]"
        )
        print(
            f"  [Epoch Summary] Max gradient norm: {epoch_stats['max_grad_norm']:.4f}"
        )
        print(
            f"  [Epoch Summary] Min entropy: {epoch_stats['min_entropy']:.4f}"
        )
        if epoch_stats["nan_batches"] > 0:
            print(
                f"  [Epoch Summary] Batches with NaN: {epoch_stats['nan_batches']}"
            )
        if epoch_stats["inf_batches"] > 0:
            print(
                f"  [Epoch Summary] Batches with Inf: {epoch_stats['inf_batches']}"
            )

        # Check model health at end of epoch
        has_nan, has_inf, weight_stats = self._check_model_health(
            "End of epoch:"
        )
        if has_nan or has_inf:
            print(
                "  [ERROR] Model has NaN/Inf weights - training has diverged!"
            )

        # Report largest weight magnitudes (potential explosion)
        max_weight_layers = sorted(
            weight_stats.items(), key=lambda x: x[1]["max"], reverse=True
        )[:3]
        print("  [Epoch Summary] Largest weight magnitudes:")
        for name, stats in max_weight_layers:
            print(
                f"      {name}: max={stats['max']:.4f}, mean={stats['mean']:.6f}"
            )

        return avg_loss

    def train(
        self,
        dataset: TransitionDataset,
        epochs: int = 10,
        batch_size: int = 64,
        save_every: int = 5,
        run_name: Optional[str] = None,
    ) -> dict:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = MODELS_FOLDER / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Pre-training diagnostics: analyze dataset
        print("\n" + "=" * 60)
        print("PRE-TRAINING DIAGNOSTICS")
        print("=" * 60)
        all_returns = torch.tensor([s.dc_return for s in dataset.samples])
        print(f"Dataset size: {len(dataset)}")
        print("Returns distribution:")
        print(f"  Mean: {all_returns.mean().item():.4f}")
        print(f"  Std:  {all_returns.std().item():.4f}")
        print(f"  Min:  {all_returns.min().item():.4f}")
        print(f"  Max:  {all_returns.max().item():.4f}")
        # Check for extreme returns that could cause issues
        extreme_threshold = 10.0
        extreme_count = (all_returns.abs() > extreme_threshold).sum().item()
        if extreme_count > 0:
            print(
                f"  [WARN] {extreme_count} samples have |return| > {extreme_threshold}"
            )
        # Check return balance
        pos_returns = (all_returns > 0).sum().item()
        neg_returns = (all_returns < 0).sum().item()
        zero_returns = (all_returns == 0).sum().item()
        print(
            f"  Positive: {pos_returns}, Negative: {neg_returns}, Zero: {zero_returns}"
        )

        # Check initial model health
        print("\nInitial model state:")
        self._check_model_health("Initial:")
        print("=" * 60 + "\n")

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(dataloader)
            self.history["loss"].append(loss)
            self.history["epoch"].append(epoch)

            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

            if epoch % save_every == 0:
                self.save_checkpoint(run_dir, epoch)

        self.save_checkpoint(run_dir, epochs, final=True)
        self._save_history(run_dir)
        return self.history

    def save_checkpoint(self, run_dir: Path, epoch: int, final: bool = False):
        suffix = "final" if final else f"epoch_{epoch:03d}"
        path = run_dir / f"model_{suffix}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        return checkpoint["epoch"]

    def _save_history(self, run_dir: Path):
        with open(run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def evaluate_vs_random(
    model: Connect4Net,
    n_games: int = 100,
    device: str = "mps",
    seed: int = 42,
) -> dict:
    """Evaluate the model against a random opponent."""
    learned = LearnedPolicy(model, temperature=0, device=device)  # greedy
    random_policy = RandomStrat(seed)

    results = {"p1_wins": 0, "p2_wins": 0, "draws": 0}

    for i in range(n_games):
        game = Game()
        player = 1
        # Alternate who goes first each game
        learned_is_p1 = i % 2 == 0

        while game.legal_moves:
            if (player == 1 and learned_is_p1) or (
                player == 2 and not learned_is_p1
            ):
                move = learned.choose_move(game, player)
            else:
                move = random_policy.choose_move(game, player)

            game.apply_move(move, player)
            if game.has_winner():
                break
            player = 2 if player == 1 else 1

        # Track results from learned policy's perspective
        if game.winner == 0:
            results["draws"] += 1
        elif (game.winner == 1 and learned_is_p1) or (
            game.winner == 2 and not learned_is_p1
        ):
            results["p1_wins"] += 1  # learned policy won
        else:
            results["p2_wins"] += 1  # random won

    results["win_rate"] = results["p1_wins"] / n_games
    results["loss_rate"] = results["p2_wins"] / n_games
    results["draw_rate"] = results["draws"] / n_games
    return results


def self_play_train(
    model: Connect4Net,
    trainer: Trainer,
    rounds: int = 100,
    games_per_round: int = 100,
    epochs_per_round: int = 1,
    batch_size: int = 64,
    temperature: float = 1.0,
    eval_every: int = 10,
    eval_games: int = 100,
    save_every: int = 20,
    run_name: Optional[str] = None,
) -> dict:
    """
    Online self-play training loop.

    The model plays against itself to generate training data,
    then trains on that data, and repeats.
    """
    run_name = (
        run_name or f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = MODELS_FOLDER / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "round": [],
        "loss": [],
        "win_rate_vs_random": [],
        "samples_per_round": [],
    }

    print(f"\n{'=' * 60}")
    print(f"SELF-PLAY TRAINING: {run_name}")
    print(f"Rounds: {rounds}, Games/round: {games_per_round}")
    print(f"{'=' * 60}\n")

    for round_idx in range(1, rounds + 1):
        # Generate data with current policy
        policy = LearnedPolicy(
            model, temperature=temperature, device=trainer.device
        )
        exp = Experiment(policy)
        exp.play(games_per_round)
        exp.build_features()

        n_samples = len(exp.samples)
        history["samples_per_round"].append(n_samples)

        # Quick stats on the self-play games
        p1_wins = sum(1 for g in exp.games if g.winner == 1)
        p2_wins = sum(1 for g in exp.games if g.winner == 2)
        draws = sum(1 for g in exp.games if g.winner == 0)
        print(
            f"Round {round_idx:3d} | Self-play: P1={p1_wins}, P2={p2_wins}, Draw={draws} | Samples: {n_samples}"
        )

        # Train on fresh data
        dataset = TransitionDataset(exp.samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs_per_round):
            loss = trainer.train_epoch(dataloader, epoch=round_idx)

        history["round"].append(round_idx)
        history["loss"].append(loss)

        print(f"           | Loss: {loss:.4f}")

        # Evaluate periodically
        if round_idx % eval_every == 0:
            eval_results = evaluate_vs_random(
                model, n_games=eval_games, device=trainer.device
            )
            history["win_rate_vs_random"].append(eval_results["win_rate"])
            print(
                f"           | vs Random: {eval_results['win_rate']:.1%} win, "
                f"{eval_results['loss_rate']:.1%} loss, {eval_results['draw_rate']:.1%} draw"
            )
        else:
            history["win_rate_vs_random"].append(None)

        # Save checkpoint periodically
        if round_idx % save_every == 0:
            trainer.save_checkpoint(run_dir, round_idx)

    # Final save
    trainer.save_checkpoint(run_dir, rounds, final=True)
    with open(run_dir / "selfplay_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    final_eval = evaluate_vs_random(model, n_games=200, device=trainer.device)
    print(f"Final vs Random (200 games): {final_eval['win_rate']:.1%} win rate")
    print(f"{'=' * 60}\n")

    return history


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Connect4Net()
    trainer = Trainer(model, lr=1e-4, device=device)

    # Self-play training
    self_play_train(
        model=model,
        trainer=trainer,
        rounds=10,
        games_per_round=100,
        epochs_per_round=1,
        batch_size=64,
        temperature=1.0,
        eval_every=1,
        eval_games=100,
        save_every=20,
        run_name="selfplay_v1",
    )


if __name__ == "__main__":
    main()
