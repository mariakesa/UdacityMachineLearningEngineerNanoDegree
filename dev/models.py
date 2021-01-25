"""
This work modifies code from the pl_bolts library.
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2018-2021 William Falcon

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
Deep Q Network
"""
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl.common.gym_wrappers import make_environment
from pl_bolts.models.rl.common.memory import MultiStepBuffer
from pl_bolts.models.rl.common.networks import CNN

from gym import Env

from abc import ABC
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from functools import partial
import collections

import torchfunc



class Agent(ABC):
    """Basic agent that always returns 0"""

    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, state: torch.Tensor, device: str, *args, **kwargs) -> List[int]:
        """
        Using the given network, decide what action to carry
        Args:
            state: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        return [0]

class ValueAgent(Agent):
    """Value based agent that returns an action based on the Q values from the network"""

    def __init__(
        self,
        net: nn.Module,
        action_space: int,
        eps_start: float = 1.0,
        eps_end: float = 0.2,
        eps_frames: float = 1000,
    ):
        super().__init__(net)
        self.action_space = action_space
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames
        self.recorder=torchfunc.hooks.recorders.ForwardPre()
        self.recorder.modules(self.net)

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str) -> List[int]:
        """
        Takes in the current state and returns the action based on the agents policy
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if not isinstance(state, list):
            state = [state]

        if np.random.random() < self.epsilon:
            action = self.get_random_action(state)
        else:
            action = self.get_action(state, device)


        return action

    def get_random_action(self, state: torch.Tensor) -> int:
        """returns a random action"""
        actions = []

        for i in range(len(state)):
            action = np.random.randint(0, self.action_space)
            actions.append(action)

        return actions

    def get_action(self, state: torch.Tensor, device: torch.device):
        """
        Returns the best action based on the Q values of the network
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by Q values
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device)

        q_values = self.net(state)

        _, actions = torch.max(q_values, dim=1)
        return actions.detach().cpu().numpy()

    def update_epsilon(self, step: int) -> None:
        """
        Updates the epsilon value based on the current step
        Args:
            step: current global step
        """
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)


class DQN(pl.LightningModule):
    """
    Basic DQN Model

    PyTorch Lightning implementation of `DQN <https://arxiv.org/abs/1312.5602>`_
    Paper authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
    Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller.
    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:
        >>> from pl_bolts.models.rl.dqn_model import DQN
        ...
        >>> model = DQN("PongNoFrameskip-v4")

    Train::

        trainer = Trainer()
        trainer.fit(model)

    Note:
        This example is based on:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py

    Note:
        Currently only supports CPU and single GPU training with `distributed_backend=dp`
    """

    def __init__(
        self,
        env: str,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        seed: int = 123,
        batches_per_epoch: int = 1000,
        n_steps: int = 1,
        **kwargs,
    ):
        """
        Args:
            env: gym environment tag
            eps_start: starting value of epsilon for the epsilon-greedy exploration
            eps_end: final value of epsilon for the epsilon-greedy exploration
            eps_last_frame: the final frame in for the decrease of epsilon. At this frame espilon = eps_end
            sync_rate: the number of iterations between syncing up the target network with the train network
            gamma: discount factor
            learning_rate: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
                training to fill the buffer with a starting point
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            min_episode_reward: the minimum score that can be achieved in an episode. Used for filling the avg buffer
                before training begins
            seed: seed value for all RNG used
            batches_per_epoch: number of batches per epoch
            n_steps: size of n step look ahead
        """
        super().__init__()

        # Environment
        self.exp = None
        self.env = self.make_environment(env, seed)
        self.test_env = self.make_environment(env)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.net = None
        self.target_net = None
        self.build_networks()

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.n_steps = n_steps

        self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))

        self.state = self.env.reset()

    def run_n_episodes(self, env, n_epsiodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """
        Carries out N episodes of the environment with the current agent

        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []
        self.im_arr=[]
        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
                action = self.agent(episode_state, self.device)
                #print(action)
                import matplotlib.pyplot as plt
                #plt.imshow(episode_state[0,:,:])
                #plt.show()
                self.im_arr.append(np.mean(episode_state,axis=0).flatten())
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)
            self.activations=self.agent.recorder.data
        self.im_arr=np.array(self.im_arr)
        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action = self.agent(self.state, self.device)
                #print(action)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=next_state)
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def train_batch(self, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)

            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({
            "total_reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
            "train_loss": loss,
            "episodes": self.done_episodes,
            "episode_steps": self.total_episode_steps[-1]
        })

        return OrderedDict({
            "loss": loss,
            "avg_reward": self.avg_rewards,
        })

    def test_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Evaluate the agent for 10 episodes"""
        test_reward = self.run_n_episodes(self.test_env, 1, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()

    @staticmethod
    def make_environment(env_name: str, seed: Optional[int] = None) -> Env:
        """
        Initialise gym  environment

        Args:
            env_name: environment name or tag
            seed: value to seed the environment RNG for reproducibility

        Returns:
            gym environment
        """
        env = make_environment(env_name)

        if seed:
            env.seed(seed)

        return env

    @staticmethod
    def add_model_specific_args(arg_parser: argparse.ArgumentParser, ) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model

        Note:
            These params are fine tuned for Pong env.

        Args:
            arg_parser: parent parser
        """
        arg_parser.add_argument(
            "--sync_rate",
            type=int,
            default=1000,
            help="how many frames do we update the target network",
        )
        arg_parser.add_argument(
            "--replay_size",
            type=int,
            default=100000,
            help="capacity of the replay buffer",
        )
        arg_parser.add_argument(
            "--warm_start_size",
            type=int,
            default=10000,
            help="how many samples do we use to fill our buffer at the start of training",
        )
        arg_parser.add_argument(
            "--eps_last_frame",
            type=int,
            default=150000,
            help="what frame should epsilon stop decaying",
        )
        arg_parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        arg_parser.add_argument("--eps_end", type=float, default=0.02, help="final value of epsilon")
        arg_parser.add_argument("--batches_per_epoch", type=int, default=10000, help="number of batches in an epoch")
        arg_parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

        arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )
        arg_parser.add_argument(
            "--n_steps",
            type=int,
            default=1,
            help="how many frames do we update the target network",
        )

        return arg_parser
