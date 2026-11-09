# Aerie

An original flight arcade and a legible, evidence-led research workspace. The distinctive visual is a folded kite-bird passing through faceted mineral gates in an atmospheric canyon. All illustration is original SVG/Canvas code in this directory. There are no remote assets or fonts.

Palette: ink #111d23, slate #1a2b32, mist #e5eee8, mint #9de6ce, coral #ee9c7f, amber #e8cb91. The arcade uses bright sky and warm rock, while the laboratory uses the same colors sparingly for measured data. Human birds use a mint round insignia; the opponent uses a coral diamond, with text labels as well as colors.

Typography: Trebuchet MS display with tight tracking; system sans body; system monospace only for identifiers and numerical data. Layout: persistent left navigation, generous page heading, then a task-specific workspace. Game scores sit outside the uniformly scaled world. Lab charts retain numeric tables for exact values and accessibility.

Review of the direction: avoid generic identical stat-card grids as the dominant composition. The home screen is a canyon illustration with a four-mode flight selector; the game emphasizes the world; the lab uses dense tables and chart panels. Motion is limited to input feedback, short state transitions, and interpolation of server snapshots. Reduced motion disables decorative parallax and feedback transforms.

Data boundary: API responses are the sole source of game state and research numbers. Empty values remain empty; fixtures are imported only in tests. No collisions, rewards, AI actions, scoring or world generation run in the browser.
