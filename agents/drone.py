import numpy as np

class Drone:
    def __init__(self, x, y, field_shape, failure_prob=0.0, controller=None):
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

    def get_position(self):
        return self.x, self.y

    def sense(self, field, pheromone_map, radius=1):
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

        if self.controller is not None:
            input_vector = self.build_input_vector(field, pheromone_map, radius)
            action = np.argmax(self.controller.forward(input_vector))
            self.apply_action(action, occupied_positions)
            return

        crop_patch, pher_patch = self.sense(field, pheromone_map, radius)
        desirability = crop_patch + pher_patch
        center = radius
        desirability[center, center] = np.inf  # discourage standing still

        # Flocking influence
        target = np.array([self.x, self.y])
        if all_positions is not None:
            flock_vec = self.compute_flocking_vector(all_positions, radius=radius, w_sep=1.0, w_coh=1.0)
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
                        score = desirability[dx + radius, dy + radius] + 0.5 * dist_to_target
                        candidates.append(((nx, ny), score))

        if not candidates:
            return  # No legal moves

        best_move = min(candidates, key=lambda x: x[1])[0]
        self.x, self.y = best_move
        self.path.append(best_move)

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

        return np.pad(patch, (pad_x, pad_y), mode='constant', constant_values=np.max(grid))

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
        pher = self._get_local_patch(pheromone_map, radius).flatten() / 5.0  # assume max pheromone ~5
        visits = self._get_local_patch(self.visit_map_ref, radius).flatten()
        visits = np.clip(visits, 0, 10) / 10.0  # normalize visit counts
        return np.concatenate([crop, pher, visits])

    def apply_action(self, action_index, occupied_positions):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # N, S, W, E, stay
        dx, dy = moves[action_index]
        new_x, new_y = self.x + dx, self.y + dy

        # Boundary + collision check
        if 0 <= new_x < self.field_shape[0] and 0 <= new_y < self.field_shape[1]:
            if (new_x, new_y) not in occupied_positions:
                self.x = new_x
                self.y = new_y
                self.path.append((self.x, self.y))
