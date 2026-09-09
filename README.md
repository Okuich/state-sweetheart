# physics OS

# state.py

class PhysicsState:

    def __init__(self, positions, velocities, mass):

        # All tensors MUST be GPU compatible

        self.x = positions        # shape: [N, D]

        self.v = velocities       # shape: [N, D]

        self.m = mass             # shape: [N]

        self.f = zeros_like(self.x)

    def to_device(self, device):

        self.x = self.x.to(device)

        self.v = self.v.to(device)

        self.m = self.m.to(device)

        self.f = self.f.to(device)

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://state-sweetheart.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/5d81cf04-63a3-4c11-bf46-7269b39cb41b).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
