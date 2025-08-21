import numpy as np

class Drone:
    def __init__(self, x, y, field_shape, visit_map, failure_prob=0.0, controller=None):
        """
        Generate the drone at position (x, y).

        Args:
            x (int): row index.
            y (int): column index.
            field_shape (tuple): (rows, cols) of the field (to check boundaries).
        """
        self.x = x
        self.y = y
        self.field_shape = field_shape
        self.path = [(x, y)]
        self.active = True
        self.failure_prob = failure_prob
        self.controller = controller
        self.visit_map_ref = visit_map
        self.steps_taken = 0
        self.MOVES = [(-1, 0),  # 0: up (N)
                 (1, 0),  # 1: down (S)
                 (0, -1),  # 2: left (W)
                 (0, 1),  # 3: right (E)
                 (0, 0)]  # 4: stay

    def get_position(self):
        return self.x, self.y

    def sense(self, field, pheromone_map, radius=2):
        """
        Look at a (2r+1)x(2r+1) area around the drone in the field and pheromone map.

        Returns:
            tuple: (local_crop, local_pheromone) as numpy arrays.
        """
        crop_patch = self._get_local_patch(field, radius)
        pheromone_patch = self._get_local_patch(pheromone_map, radius)

        return crop_patch, pheromone_patch

    def decide_and_move(self, field, pheromone_map, occupied_positions, all_positions=None, radius=1):
        """
        Choose the next cell to move to (based on crop health and pheromone):
        Move to the neighboring cell with the lowest (pheromone + crop health) score.
        """
        if not self.active:
            return

        # e-greedy exploration
        epsilon_start = 0.02
        epsilon_end = 0.001
        decay_rate = 0.998
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (decay_rate ** self.steps_taken)

        if self.controller is not None:
            x = self.build_input_vector(field, pheromone_map, radius)
            nn = self.controller.forward(x).astype(float)  # raw NN logits
            mask = self.valid_action_mask(occupied_positions)

            if np.random.rand() < epsilon:
                valid_idxs = np.flatnonzero(mask)
                if valid_idxs.size == 0:
                    self.apply_action(4, occupied_positions)
                    self.steps_taken += 1
                    return
                if 4 in valid_idxs and valid_idxs.size > 1:
                    valid_idxs = valid_idxs[valid_idxs != 4]
                action = int(np.random.choice(valid_idxs))
                self.apply_action(action, occupied_positions)
                self.steps_taken += 1
                return

            if len(self.path) >= 2:
                px, py = self.path[-2]
                back_a = None
                for a, (dx, dy) in enumerate(self.MOVES[:4]):  # only movement actions
                    nx, ny = self.x + dx, self.y + dy
                    if (nx, ny) == (px, py):
                        back_a = a
                        break
                if back_a is not None and np.sum(mask[:4]) > 1:
                    mask[back_a] = False

            H, W = self.field_shape
            local_pher = np.full(5, 0.0)
            wall_bias = np.full(5, 0.0)
            poor_crop = np.full(5, 0.0)
            low_visit = np.full(5, 0.0)  # higher = less visited (what we want)

            # global crop scale (matches evaluator’s stress)
            fmin, fmax = field.min(), field.max()
            # local pher scale
            p_patch = self._get_local_patch(pheromone_map, radius)
            pmin, pmax = p_patch.min(), p_patch.max()
            pden = (pmax - pmin + 1e-6)

            for a, (dx, dy) in enumerate(self.MOVES):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    health = (field[nx, ny] - fmin) / (fmax - fmin + 1e-9)
                    poor_crop[a] = 1.0 - health  # prefer sick
                    local_pher[a] = (pheromone_map[nx, ny] - pmin) / pden
                    wall_bias[a] = min(nx, ny, H - 1 - nx, W - 1 - ny) / max(1, min(H, W) // 2)
                    v = self.visit_map_ref[nx, ny]
                    low_visit[a] = 1.0 / (1.0 + v)

            # standardize over valid actions so weights are comparable
            def z(v):
                vm = v[mask]
                m = vm.mean() if vm.size else 0.0
                s = vm.std() if vm.size else 1.0
                return (v - m) / (s + 1e-6)

            score = (
                    0.7 * z(nn)  # learned preference
                    + 1.2 * z(low_visit)
                    + 0.8 * z(poor_crop)
                    - 0.6 * z(local_pher)
                    + 0.2 * z(wall_bias)
            )

            if all_positions is not None:
                sep = np.zeros(5)
                for a, (dx, dy) in enumerate(self.MOVES[:4]):  # exclude stay
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        for px, py in all_positions:
                            if (px, py) != (self.x, self.y) and abs(px - nx) <= 1 and abs(py - ny) <= 1:
                                sep[a] += 1.0
                score = score - 0.1 * z(sep)

            # pick action
            score = np.where(mask, score, -1e9)
            score = score + 1e-6 * np.random.randn(*score.shape)  # tie-break
            action = int(np.argmax(score))

            self.apply_action(action, occupied_positions)
            self.steps_taken += 1
            return

        crop_patch, pher_patch = self.sense(field, pheromone_map, radius)
        visit_patch = self._get_local_patch(self.visit_map_ref, radius)
        pher_norm = pher_patch / (np.max(pheromone_map) + 1e-5)
        visit_penalty = np.log1p(visit_patch) / np.log1p(np.max(self.visit_map_ref) + 1e-5)
        anti_pher = np.clip(1.0 - pher_norm, 0.0, 1.0)

        coverage_ratio = np.count_nonzero(self.visit_map_ref) / self.visit_map_ref.size
        explore_pressure = 1.0 - coverage_ratio

        row_visits = np.sum(self.visit_map_ref, axis=1)
        row_bonus = 1.0 - (row_visits[self.x] / (np.max(row_visits) + 1e-5))
        col_visits = np.sum(self.visit_map_ref, axis=0)
        col_bonus = 1.0 - (col_visits[self.y] / (np.max(col_visits) + 1e-5))
        explore_bonus = (1.0 - visit_penalty) * anti_pher

        w_crop, w_pher, w_visit, w_explore, w_row, w_col = 1.0, 1.5, 1.5, 5.0, 0.8, 0.8
        desirability = (
                - w_crop * crop_patch
                - w_pher * pher_norm
                - w_visit * visit_penalty
                + w_explore * explore_bonus * explore_pressure
                + w_row * row_bonus
                + w_col * col_bonus
        )
        center = radius
        desirability[center, center] = -np.inf  # discourage standing still

        # Flocking influence
        target = np.array([self.x, self.y])
        if all_positions is not None:
            flock_vec = self.compute_flocking_vector(all_positions, radius=radius, w_sep=5.0, w_coh=0.0)
            target = target + np.round(flock_vec).astype(int)
            target[0] = np.clip(target[0], 0, self.field_shape[0] - 1)
            target[1] = np.clip(target[1], 0, self.field_shape[1] - 1)

        # Scoring candidates
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < self.field_shape[0] and 0 <= ny < self.field_shape[1]:
                    if (nx, ny) not in occupied_positions:
                        dist_to_target = np.linalg.norm([nx - target[0], ny - target[1]])
                        score = desirability[dx + radius, dy + radius] - dist_to_target
                        candidates.append(((nx, ny), score))

        if not candidates:
            return  # No legal moves

        best_move = max(candidates, key=lambda x: x[1])[0]
        self.x, self.y = best_move
        self.path.append(best_move)
        self.steps_taken += 1

    def deposit_pheromone(self, pheromone_map_obj, amount=1.0):
        """
        Leave pheromone at current position (x, y).
        """
        if not self.active:
            return
        pheromone_map_obj.deposit(self.x, self.y, amount)

    def _get_local_patch(self, grid, radius):
        """
        Get a local patch centered on current position.

        Args:
            grid (np.ndarray): Field or pheromone map
            radius (int): Neighborhood radius

        Returns:
            np.ndarray: patch centered on (x, y)
        """
        x_min = max(self.x - radius, 0)
        x_max = min(self.x + radius + 1, grid.shape[0])
        y_min = max(self.y - radius, 0)
        y_max = min(self.y + radius + 1, grid.shape[1])

        patch = grid[x_min:x_max, y_min:y_max]

        # Pad if near edges to keep shape consistent
        pad_x = (max(0, radius - self.x), max(0, self.x + radius + 1 - grid.shape[0]))
        pad_y = (max(0, radius - self.y), max(0, self.y + radius + 1 - grid.shape[1]))

        return np.pad(patch, (pad_x, pad_y), mode='edge')

    def compute_flocking_vector(self, all_positions, radius=2, w_sep=1.0, w_coh=1.0):
        """
        Compute flocking vector based on neighboring drone positions.

        Args:
            all_positions (list): List of (x, y) positions of all drones.
            radius (int): Sensing radius.
            w_sep (float): Weight for separation.
            w_coh (float): Weight for cohesion.

        Returns:
            np.ndarray: Flocking direction as a vector (dx, dy).
        """
        neighbors = []
        for pos in all_positions:
            if pos != (self.x, self.y):
                dx = pos[0] - self.x
                dy = pos[1] - self.y
                if abs(dx) <= radius and abs(dy) <= radius:
                    neighbors.append(pos)

        sep_vec = np.array([0.0, 0.0])
        coh_vec = np.array([0.0, 0.0])

        for nx, ny in neighbors:
            diff = np.array([self.x - nx, self.y - ny])
            sep_vec += diff / (np.linalg.norm(diff) + 1e-5)
            coh_vec += np.array([nx, ny])

        if neighbors:
            coh_vec = coh_vec / len(neighbors) - np.array([self.x, self.y])

        flock_vec = w_sep * sep_vec + w_coh * coh_vec
        return flock_vec

    def maybe_fail(self):
        """
        Randomly deactivate the drone based on failure probability.
        """
        if self.active and np.random.rand() < self.failure_prob:
            self.active = False

    def build_input_vector(self, field, pheromone_map, radius):
        crop = self._get_local_patch(field, radius).flatten() / 2.0  # crop health: 0–2
        pher = 1- self._get_local_patch(pheromone_map, radius).flatten() / 5.0
        visits = self._get_local_patch(self.visit_map_ref, radius).flatten()
        visits = 1 - np.clip(visits, 0, 10) / 10.0  # normalize visit counts
        pos_info = np.array([self.x / self.field_shape[0], self.y / self.field_shape[1]])

        return np.concatenate([crop, pher, visits, pos_info])

    def valid_action_mask(self, occupied_positions):
        H, W = self.field_shape
        mask = np.zeros(5, dtype=bool)

        movable = False
        for a, (dx, dy) in enumerate(self.MOVES[:4]):
            nx, ny = self.x + dx, self.y + dy
            ok = (0 <= nx < H) and (0 <= ny < W) and ((nx, ny) not in occupied_positions)
            mask[a] = ok
            movable |= ok

        # allow stay only if nothing else is possible
        mask[4] = not movable
        return mask

    def apply_action(self, action_index, occupied_positions):
        dx, dy = self.MOVES[action_index]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < self.field_shape[0] and 0 <= ny < self.field_shape[1]):
            self.path.append((self.x, self.y))
            return

        if (dx, dy) == (0, 0):
            pass
        else:
            if (nx, ny) in occupied_positions:
                self.path.append((self.x, self.y))
                return

        self.x, self.y = nx, ny
        self.path.append((self.x, self.y))

    # def apply_action(self, action_index, occupied_positions):
    #     moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # N, S, W, E, stay
    #     dx, dy = moves[action_index]
    #     new_x, new_y = self.x + dx, self.y + dy
    #
    #     # Boundary + collision check
    #     if 0 <= new_x < self.field_shape[0] and 0 <= new_y < self.field_shape[1]:
    #         if (new_x, new_y) not in occupied_positions:
    #             self.x = new_x
    #             self.y = new_y
    #             self.path.append((self.x, self.y))
    #
    # def valid_action_mask(self, occupied_positions):
    #     moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    #     mask = []
    #     for dx, dy in moves:
    #         nx, ny = self.x + dx, self.y + dy
    #         valid = (0 <= nx < self.field_shape[0] and
    #                  0 <= ny < self.field_shape[1] and
    #                  (nx, ny) not in occupied_positions)
    #         mask.append(valid)
    #     return np.array(mask, dtype=bool)

