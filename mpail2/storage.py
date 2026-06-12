import math
import torch


class DictDemoDataset(torch.utils.data.Dataset):
    def __init__(self, demo_dict, horizon=1, device='cuda', obs_normalizer=None):
        """
        Dataset for sampling consecutive subtrajectories from flattened expert demonstrations.

        Args:
            demo_dict: Dictionary of demonstrations with shape [N, 2, *obs_shape]
                      where N is flattened trajectories and 2 is (obs, next_obs)
            horizon: Length of subtrajectories to sample
            device: Device for sampling operations
            obs_normalizer: Optional observation normalizer to apply to demonstrations
        """
        self.demo_dict = demo_dict
        self.horizon = horizon
        self.device = device
        self.obs_normalizer = obs_normalizer

        # All groups should have same N
        first_value = next(iter(demo_dict.values()))
        self.num_transitions = first_value.shape[0]

        # Calculate number of valid subtrajectories
        # We can sample starting from indices 0 to (num_transitions - horizon)
        self.num_samples = max(1, self.num_transitions - horizon + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return a consecutive subtrajectory of length horizon starting at idx.

        Returns:
            obs_traj: Dictionary {group: [horizon, *obs_shape]}
            next_obs_traj: Dictionary {group: [horizon, *obs_shape]}
        """
        # Sample consecutive transitions from idx to idx+horizon
        end_idx = idx + self.horizon

        obs_traj = {group: data[idx:end_idx, 0] for group, data in self.demo_dict.items()}
        next_obs_traj = {group: data[idx:end_idx, 1] for group, data in self.demo_dict.items()}

        # Apply observation normalization if available
        if self.obs_normalizer is not None:
            obs_traj = self.obs_normalizer(obs_traj)
            next_obs_traj = self.obs_normalizer(next_obs_traj)

        return obs_traj, next_obs_traj

    def mini_traj_batch_generator(self, batch_size, num_epochs=1):
        """
        Generate batches of subtrajectories, sampling without replacement within each epoch.

        Args:
            batch_size: Number of subtrajectories per batch
            num_epochs: Number of epochs to iterate

        Yields:
            Tuple of (obs_batch, next_obs_batch)
            Shape of each batch element: (batch_size, horizon, *feature_dims)
        """
        if self.num_samples == 0:
            for _ in range(num_epochs):
                yield (None, None)
            return

        # Clamp batch size to available samples
        actual_batch_size = min(batch_size, self.num_samples)

        for epoch in range(num_epochs):
            if actual_batch_size <= 0:
                yield (None, None)
                continue

            # Sample unique start indices without replacement using direct index computation
            sample_indices = torch.randperm(self.num_samples, device=self.device)[:actual_batch_size]

            # Build sequences using tensor indexing
            # Create time indices: (horizon, batch_size) where each column is start_idx + [0,1,2,...,horizon-1]
            time_indices = sample_indices.unsqueeze(0) + torch.arange(self.horizon, device=self.device).unsqueeze(1)

            # Index into demonstrations: [N, 2, *obs_shape]
            # time_indices has shape [horizon, batch_size], use advanced indexing
            obs_batch = {k: v[time_indices, 0].transpose(0, 1) for k, v in self.demo_dict.items()}
            next_obs_batch = {k: v[time_indices, 1].transpose(0, 1) for k, v in self.demo_dict.items()}

            # Apply observation normalization if available
            if self.obs_normalizer is not None:
                obs_batch = self.obs_normalizer(obs_batch)
                next_obs_batch = self.obs_normalizer(next_obs_batch)

            # Result shape: [batch_size, horizon, *obs_shape]
            yield (obs_batch, next_obs_batch)


class RolloutStorage:
    """Storage for MPAIL training with dictionary observations.

    Maintains episode storage and a replay buffer for off-policy learning.
    Uses stack (FIFO) replacement strategy for the replay buffer.
    """

    class Transition:
        """Container for environment transitions."""
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.valid = None

        def clear(self):
            self.__init__()

    def __init__(self,
        num_envs,
        num_steps_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        replay_size=None,
        replay_batch_size=None,
        device=torch.device("cuda"),
        obs_normalizer=None,
        protect_demo_replay=False,
    ):
        # Store basic parameters
        self.device = device
        self.num_steps_per_env = num_steps_per_env
        self.num_envs = num_envs
        self.action_shape = action_shape
        self.replay_batch_size = replay_batch_size
        self.protect_demo_replay = protect_demo_replay

        # Observations are always dictionaries
        self.actor_obs_shape = actor_obs_shape

        # Initialize episode storage
        self.observations = {
            k: torch.zeros(num_steps_per_env, num_envs, *v, device=self.device)
            for k, v in actor_obs_shape.items()
        }

        self.actions = torch.zeros(num_steps_per_env, num_envs, *action_shape, device=self.device)
        self.rewards = torch.zeros(num_steps_per_env, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps_per_env, num_envs, 1, device=self.device).byte()
        self.valid = torch.ones(num_steps_per_env, num_envs, 1, device=self.device).byte()

        # Privileged observations (not used in MPAIL but kept for compatibility)
        self.privileged_observations = None

        # Initialize replay buffer
        self._replay = replay_size is not None
        if self._replay:

            self.replay_dim_size = math.ceil(replay_size / num_steps_per_env) # number of trajectories

            self.replay_data = {
                "obs": {k: torch.zeros(num_steps_per_env, self.replay_dim_size, *v, device=self.device)
                        for k, v in actor_obs_shape.items()},
                "actions": torch.zeros(num_steps_per_env, self.replay_dim_size, *action_shape, device=self.device),
                "dones": torch.zeros(num_steps_per_env, self.replay_dim_size, 1, device=self.device),
                "valid": torch.ones(num_steps_per_env, self.replay_dim_size, 1, device=self.device),
                "is_demo": torch.zeros(num_steps_per_env, self.replay_dim_size, 1, device=self.device),
                "age": torch.zeros(self.replay_dim_size, device=self.device),
            }

        self.num_trajectories_stored = 0
        self.obs_normalizer = obs_normalizer

        # Counter for number of transitions stored
        self.step = 0

    def add_transitions(self, transition: 'RolloutStorage.Transition'):

        # check if the transition is valid
        if self.step >= self.num_steps_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core - observations are always dict
        for k in self.actor_obs_shape.keys():
            self.observations[k][self.step].copy_(transition.observations[k])

        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        if transition.valid is None:
            self.valid[self.step].fill_(1)
        else:
            self.valid[self.step].copy_(transition.valid.view(-1, 1).to(torch.uint8))

        self.step += 1

    def clear(self):
        """Clear the episode storage."""
        self.step = 0

    def _get_obs(self, obs_storage, key=None):
        """Get observations from dict storage"""
        if key is not None:
            return obs_storage[key]
        # Return the first key's observations for shape inference
        return next(iter(obs_storage.values()))

    def _get_num_envs(self):
        """Get number of envs from dict observations"""
        return self._get_obs(self.observations).shape[1]

    def _roll_obs(self, obs_storage):
        """Roll observations to get next_obs"""
        return {k: torch.roll(v, shifts=-1, dims=0) for k, v in obs_storage.items()}

    @property
    def has_replay_data(self):
        """Check if replay buffer has sufficient data for training"""
        if not self._replay:
            return False
        # Check if we have at least some data in the replay buffer
        return self.num_trajectories_stored > 0

    def _increment_replay_age(self):
        if self.num_trajectories_stored > 0:
            self.replay_data["age"][:self.num_trajectories_stored] += 1

    def _add_new_data_to_buffer(self):
        """Helper to add new trajectory data to replay buffer.
        Returns the number of samples that were added (for overflow handling)."""
        num_envs = self._get_num_envs()
        space_remaining = self.replay_dim_size - self.num_trajectories_stored

        if space_remaining <= 0:
            return 0  # Buffer already full, no samples added

        # Only add as much as will fit
        num_to_add = min(num_envs, space_remaining)
        end_indx = self.num_trajectories_stored + num_to_add

        obs = self.observations
        actions = self.actions
        dones = self.dones
        valid = self.valid

        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, self.num_trajectories_stored:end_indx] = obs[k][:, :num_to_add]

        self.replay_data["actions"][:, self.num_trajectories_stored:end_indx] = actions[:, :num_to_add]
        self.replay_data["dones"][:, self.num_trajectories_stored:end_indx] = dones[:, :num_to_add].to(torch.float32)
        self.replay_data["valid"][:, self.num_trajectories_stored:end_indx] = valid[:, :num_to_add].to(torch.float32)
        self.replay_data["is_demo"][:, self.num_trajectories_stored:end_indx].zero_()
        self.replay_data["age"][self.num_trajectories_stored:end_indx] = 0

        self.num_trajectories_stored = end_indx

        # Return number of samples added
        return num_to_add

    def process_replay_buffer(self):
        """Main entry point for processing replay buffer using FIFO strategy."""
        if not self._replay:
            return

        self._increment_replay_age()

        # Try to add new data to buffer first
        num_added = self._add_new_data_to_buffer()

        # If buffer has space and we added all data, we're done
        if num_added == self._get_num_envs():
            return

        # Buffer is full - use FIFO to add remaining data
        num_envs = self._get_num_envs()
        num_samples = num_envs - num_added
        if num_samples <= 0:
            return

        # Shift existing data left to make room for new data
        shift_size = num_samples
        if self.protect_demo_replay and self.num_trajectories_stored > 0:
            demo_cols = self.replay_data["is_demo"][:, :self.num_trajectories_stored, 0].bool().any(dim=0)
            non_demo_cols = (~demo_cols).nonzero(as_tuple=False).flatten()
            if non_demo_cols.numel() >= shift_size:
                drop_cols = non_demo_cols[:shift_size]
                keep_mask = torch.ones(self.num_trajectories_stored, device=self.device, dtype=torch.bool)
                keep_mask[drop_cols] = False
                keep_cols = keep_mask.nonzero(as_tuple=False).flatten()
                num_kept = min(keep_cols.numel(), self.replay_dim_size - num_samples)
                keep_cols = keep_cols[:num_kept]
                for k in self.actor_obs_shape.keys():
                    self.replay_data["obs"][k][:, :num_kept] = self.replay_data["obs"][k][:, keep_cols].clone()
                self.replay_data["actions"][:, :num_kept] = self.replay_data["actions"][:, keep_cols].clone()
                self.replay_data["dones"][:, :num_kept] = self.replay_data["dones"][:, keep_cols].clone()
                self.replay_data["valid"][:, :num_kept] = self.replay_data["valid"][:, keep_cols].clone()
                self.replay_data["is_demo"][:, :num_kept] = self.replay_data["is_demo"][:, keep_cols].clone()
                self.replay_data["age"][:num_kept] = self.replay_data["age"][keep_cols].clone()

                obs = self.observations
                actions = self.actions
                dones = self.dones
                valid = self.valid
                start_idx = num_kept
                end_idx = start_idx + num_samples
                for k in self.actor_obs_shape.keys():
                    self.replay_data["obs"][k][:, start_idx:end_idx] = obs[k][:, num_added:num_added + num_samples]
                self.replay_data["actions"][:, start_idx:end_idx] = actions[:, num_added:num_added + num_samples]
                self.replay_data["dones"][:, start_idx:end_idx] = dones[:, num_added:num_added + num_samples].to(torch.float32)
                self.replay_data["valid"][:, start_idx:end_idx] = valid[:, num_added:num_added + num_samples].to(torch.float32)
                self.replay_data["is_demo"][:, start_idx:end_idx].zero_()
                self.replay_data["age"][start_idx:end_idx] = 0
                self.num_trajectories_stored = min(self.replay_dim_size, end_idx)
                return

        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, :-shift_size] = self.replay_data["obs"][k][:, shift_size:].clone()
        self.replay_data["actions"][:, :-shift_size] = self.replay_data["actions"][:, shift_size:].clone()
        self.replay_data["dones"][:, :-shift_size] = self.replay_data["dones"][:, shift_size:].clone()
        self.replay_data["valid"][:, :-shift_size] = self.replay_data["valid"][:, shift_size:].clone()
        self.replay_data["is_demo"][:, :-shift_size] = self.replay_data["is_demo"][:, shift_size:].clone()
        self.replay_data["age"][:-shift_size] = self.replay_data["age"][shift_size:].clone()

        # Add new data at the end
        obs = self.observations
        actions = self.actions
        dones = self.dones
        valid = self.valid

        start_idx = self.replay_dim_size - num_samples
        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, start_idx:] = obs[k][:, num_added:num_added + num_samples]
        self.replay_data["actions"][:, start_idx:] = actions[:, num_added:num_added + num_samples]
        self.replay_data["dones"][:, start_idx:] = dones[:, num_added:num_added + num_samples].to(torch.float32)
        self.replay_data["valid"][:, start_idx:] = valid[:, num_added:num_added + num_samples].to(torch.float32)
        self.replay_data["is_demo"][:, start_idx:].zero_()
        self.replay_data["age"][start_idx:] = 0

    def add_demo_transitions_to_replay(
        self,
        obs,
        next_obs,
        actions,
        not_done,
        max_trajectories=None,
    ):
        """Seed replay with expert transitions and mark them as demo-source data."""
        if not self._replay or obs is None or actions is None or actions.shape[0] == 0:
            return 0

        max_cols = self.replay_dim_size - self.num_trajectories_stored
        if max_trajectories is not None:
            max_cols = min(max_cols, int(max_trajectories))
        if max_cols <= 0:
            return 0

        num_transitions = actions.shape[0]
        traj_len = self.num_steps_per_env
        max_count = max(1, traj_len - 1)
        not_done_flat = not_done.float().view(-1)

        # Preserve demonstration episode boundaries when seeding replay.  The
        # clean OmniReset demos are a flattened list of episodes; packing them
        # as one long stream can create replay columns that cross from the
        # terminal transition of one episode into the reset state of the next.
        segments = []
        start = 0
        done_indices = (not_done_flat[:num_transitions] <= 0.5).nonzero(as_tuple=False).flatten()
        for done_idx in done_indices.tolist():
            end = int(done_idx) + 1
            if end > start:
                segments.append((start, end))
            start = end
        if start < num_transitions:
            segments.append((start, num_transitions))
        if not segments:
            segments.append((0, num_transitions))

        chunks = []
        for seg_start, seg_end in segments:
            cursor = seg_start
            while cursor < seg_end:
                chunk_end = min(cursor + max_count, seg_end)
                chunks.append((cursor, chunk_end))
                cursor = chunk_end
        if not chunks:
            return 0

        added = 0
        chunk_cursor = 0
        while added < max_cols:
            chunk_start, chunk_end = chunks[chunk_cursor % len(chunks)]
            count = chunk_end - chunk_start
            if count <= 0:
                break
            col = self.num_trajectories_stored + added

            for k in self.actor_obs_shape.keys():
                self.replay_data["obs"][k][:, col].zero_()
            self.replay_data["actions"][:, col].zero_()
            self.replay_data["dones"][:, col].fill_(1.0)
            self.replay_data["valid"][:, col].zero_()
            self.replay_data["is_demo"][:, col].zero_()

            idx = torch.arange(chunk_start, chunk_end, device=self.device)
            for k in self.actor_obs_shape.keys():
                if k not in obs:
                    continue
                self.replay_data["obs"][k][:count, col] = obs[k][idx]
                self.replay_data["obs"][k][1:count + 1, col] = next_obs[k][idx]
            self.replay_data["actions"][:count, col] = actions[idx]
            dones = (1.0 - not_done[idx].float()).view(-1, 1)
            dones[-1] = 1.0
            self.replay_data["dones"][:count, col] = dones
            self.replay_data["valid"][:count, col].fill_(1.0)
            self.replay_data["is_demo"][:count, col].fill_(1.0)
            self.replay_data["age"][col] = 0

            chunk_cursor += 1
            added += 1

        self.num_trajectories_stored += added
        return added

    def compute_replay_stats(self):
        if not self._replay or self.num_trajectories_stored == 0:
            return {}
        ages = self.replay_data["age"][:self.num_trajectories_stored]
        return {
            "Storage/replay_age_min": ages.min().item(),
            "Storage/replay_age_mean": ages.float().mean().item(),
            "Storage/replay_age_max": ages.max().item(),
            "Storage/replay_capacity_trajs": float(self.replay_dim_size),
            "Storage/replay_stored_trajs": float(self.num_trajectories_stored),
            "Storage/replay_demo_trajs": float(
                self.replay_data["is_demo"][:, :self.num_trajectories_stored, 0].bool().any(dim=0).sum().item()
            ),
        }

    def _compute_valid_starts(self, dones_data, horizon, max_start, valid_data=None):
        """
        Compute valid starting positions for trajectory sampling that don't cross episode boundaries.

        A start position t in trajectory i is valid if:
        - dones[t, i], dones[t+1, i], ..., dones[t+H-1, i] are all 0

        IsaacLab auto-resets done environments before returning observations from
        step().  This means the next observation after a done belongs to the next
        episode, not the terminal state.  We therefore exclude done transitions
        themselves instead of allowing terminal transitions at the end.

        Args:
            dones_data: [T, num_trajs, 1] tensor of done flags
            horizon: Length of subsequences to sample
            max_start: Maximum valid start index (T - horizon - 1)

        Returns:
            valid_indices: [N, 2] tensor of (traj_idx, start_time) pairs, or None if all valid
        """
        T, num_trajs, _ = dones_data.shape
        dones = dones_data.squeeze(-1)  # [T, num_trajs]
        valid = None if valid_data is None else valid_data.squeeze(-1).bool()

        # We need to check that dones[t:t+H] has no 1s for each start position t.
        # Using efficient cumsum approach:
        # - cumsum gives running count of dones
        # - For window [t, t+H-1), count = cumsum[t+H-1] - cumsum[t] (with t=0 edge case)

        # Compute cumsum along time dimension
        cumsum = torch.cumsum(dones.float(), dim=0)  # [T, num_trajs]

        # Pad with zeros at the start for easier indexing
        padded = torch.cat([torch.zeros(1, num_trajs, device=self.device), cumsum], dim=0)  # [T+1, num_trajs]

        # For start position t, window is [t, t+H] (H elements, including the
        # last transition in the sampled horizon).
        num_valid_starts = max_start + 1

        # Compute done counts for each window [t, t+H) for t in [0, max_start].
        # window_dones[t, i] = number of dones in dones[t:t+H, i]
        end_indices = torch.arange(horizon, horizon + num_valid_starts, device=self.device)
        start_indices = torch.arange(num_valid_starts, device=self.device)

        window_dones = padded[end_indices, :] - padded[start_indices, :]  # [num_valid_starts, num_trajs]

        # A position is valid if window has no dones (count == 0)
        valid_mask = (window_dones == 0)  # [num_valid_starts, num_trajs]
        if valid is not None:
            valid_cumsum = torch.cumsum((~valid).float(), dim=0)
            padded_invalid = torch.cat([torch.zeros(1, num_trajs, device=self.device), valid_cumsum], dim=0)
            window_invalid = padded_invalid[end_indices, :] - padded_invalid[start_indices, :]
            valid_mask &= (window_invalid == 0)

        # Check if all positions are valid (common case for fixed-length episodes)
        if valid_mask.all():
            return None  # Signal that all positions are valid

        # Get indices of valid (start_time, traj_idx) pairs
        valid_positions = valid_mask.nonzero(as_tuple=False)  # [N, 2] with (start_time, traj_idx)

        # Swap columns to get (traj_idx, start_time) format
        valid_indices = valid_positions[:, [1, 0]]  # [N, 2] with (traj_idx, start_time)

        return valid_indices

    def mini_traj_batch_generator(self, horizon, num_epochs=8, return_demo_mask=False):
        """Generate batches of trajectory subsequences from replay buffer.

        Only samples subsequences that stay within episode boundaries (don't cross dones).
        This ensures all sampled data is valid for dynamics and value learning.

        Args:
            horizon: Length of trajectory subsequences to sample
            num_epochs: Number of epochs to iterate over the data
        Yields:
            Tuple of (obs_batch, actions_batch, next_obs_batch, terminal_mask)
            - obs_batch, actions_batch, next_obs_batch: Shape (batch_size, horizon, feature_dims)
            - terminal_mask: Shape (batch_size,) - True if last timestep is a terminal state
        """

        if not self._replay or self.num_trajectories_stored == 0:
            # No replay data available, yield empty batches
            for epoch in range(num_epochs):
                if return_demo_mask:
                    yield (None, None, None, None, None)
                else:
                    yield (None, None, None, None)
            return

        # Keep reference to replay data without materializing large slices
        # This avoids allocating huge tensors for image observations
        replay_obs_data = self.replay_data["obs"]
        replay_actions_traj = self.replay_data["actions"][:, :self.num_trajectories_stored]
        replay_dones_traj = self.replay_data["dones"][:, :self.num_trajectories_stored]
        replay_valid_traj = self.replay_data["valid"][:, :self.num_trajectories_stored]
        replay_is_demo_traj = self.replay_data["is_demo"][:, :self.num_trajectories_stored]

        trajectory_length = self.num_steps_per_env
        num_trajectories = self.num_trajectories_stored

        # Clamp horizon to available trajectory length
        horizon = min(horizon, trajectory_length)

        # Determine batch size (number of subsequences to sample per epoch)
        if self.replay_batch_size is not None:
            batch_size = self.replay_batch_size
        else:
            batch_size = num_trajectories

        # Calculate max start index
        # Exclude last timestep since it has invalid next_obs (wraps to first)
        max_start = trajectory_length - horizon - 1
        if max_start < 0:
            max_start = 0

        # Compute valid start positions that don't cross episode boundaries
        valid_indices = self._compute_valid_starts(replay_dones_traj, horizon, max_start, replay_valid_traj)

        if valid_indices is None:
            # All positions are valid - use original flat indexing (more efficient)
            num_starts_per_traj = max_start + 1
            max_subsequences = num_trajectories * num_starts_per_traj
            use_valid_indices = False
        else:
            # Only some positions are valid - sample from valid_indices
            max_subsequences = valid_indices.shape[0]
            use_valid_indices = True

        if max_subsequences == 0:
            # No valid subsequences (can happen if all trajectories have early terminations)
            for epoch in range(num_epochs):
                if return_demo_mask:
                    yield (None, None, None, None, None)
                else:
                    yield (None, None, None, None)
            return

        # Determine actual batch size based on available subsequences
        actual_batch_size = min(batch_size, max_subsequences)

        for epoch in range(num_epochs):
            if actual_batch_size <= 0:
                if return_demo_mask:
                    yield (None, None, None, None, None)
                else:
                    yield (None, None, None, None)
                continue

            # Sample unique subsequences without replacement
            sample_indices = torch.randperm(max_subsequences, device=self.device)[:actual_batch_size]

            if use_valid_indices:
                # Index into precomputed valid (traj_idx, start_time) pairs
                sampled_pairs = valid_indices[sample_indices]  # [batch, 2]
                traj_indices = sampled_pairs[:, 0]
                start_times = sampled_pairs[:, 1]
            else:
                # Convert flat indices to (traj_idx, start_time) using modular arithmetic
                num_starts_per_traj = max_start + 1
                traj_indices = sample_indices // num_starts_per_traj
                start_times = sample_indices % num_starts_per_traj

            # Build sequences using tensor indexing
            # Create time indices: (horizon, batch) where each column is start_time + [0,1,2,...,horizon-1]
            time_indices = start_times.unsqueeze(0) + torch.arange(horizon, device=self.device).unsqueeze(1)
            # For next_obs, shift time indices by 1 (avoids expensive torch.roll on full buffer)
            next_time_indices = time_indices + 1

            # Index directly from replay data - only allocates batch_size elements, not full buffer
            obs_seq = {k: v[time_indices, traj_indices].transpose(0, 1)
                       for k, v in replay_obs_data.items()}
            next_obs_seq = {k: v[next_time_indices, traj_indices].transpose(0, 1)
                           for k, v in replay_obs_data.items()}

            # Apply observation normalization on the batch (not the full buffer)
            if self.obs_normalizer is not None:
                obs_seq = self.obs_normalizer(obs_seq)
                next_obs_seq = self.obs_normalizer(next_obs_seq)

            actions_seq = replay_actions_traj[time_indices, traj_indices].transpose(0, 1)

            # Get terminal mask for the last timestep of each subsequence
            # This tells the learner if the last transition ends in a terminal state
            # Shape: [batch_size] - True if done[t+H-1] == 1
            last_time_indices = time_indices[-1]  # [batch]
            terminal_mask = replay_dones_traj[last_time_indices, traj_indices, 0].bool()  # [batch]

            if return_demo_mask:
                demo_mask = replay_is_demo_traj[time_indices, traj_indices].transpose(0, 1)
                yield (obs_seq, actions_seq, next_obs_seq, terminal_mask, demo_mask)
            else:
                yield (obs_seq, actions_seq, next_obs_seq, terminal_mask)
