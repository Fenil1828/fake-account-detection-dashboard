# Sample Generator & Manual Account Entry

## Features

### 🚀 Quick Generate
- **Gen 50**: Generate 50 realistic sample accounts (mix of bot and real patterns)
- **Gen 100**: Generate 100 realistic sample accounts
- Automatic distribution: ~40% bots, ~60% real users

### ✍️ Manual Entry
Beautiful form to add individual accounts with:

#### Basic Info Section
- Username *required
- Bio
- Location

#### Audience Metrics Section
- Followers *required
- Following *required

#### Activity Metrics Section
- Total Posts
- Age (Days) *required

#### Auto-calculated Metrics
Live display showing:
- **Posts/Day**: Automatically calculated from Total Posts ÷ Age
- **Follow Ratio**: Automatically calculated from Following ÷ Followers

#### Profile Flags
- ☑️ Has Profile Image
- ☑️ Verified

### 📊 Activity Metrics Display Example

```
Posts/Day: 750.0              Follow Ratio: 1600.00x
─────────────────────────────────────────────────
Posting Frequency             Following/Followers
─────────────────────────────────────────────────
Total Posts: 15,000           Followers: 5
Lifetime Posts                Audience Size
─────────────────────────────────────────────────
Age: 20 Days Active
```

## Design Features

✨ **Beautiful & Minimal**
- Glass-morphism modal with smooth animations
- Gradient buttons with hover effects
- Color-coded input feedback
- Responsive on all screen sizes

🎨 **UI Elements**
- Mode tabs (Bulk Generate / Manual Entry)
- Real-time metric calculation
- Visual feedback on all interactions
- Smooth transitions and animations

## How to Use

1. Click **Add Account** button (purple gradient)
2. Choose mode:
   - **Bulk Generate**: Click 50 or 100 for instant generation
   - **Manual Entry**: Fill form fields and click "Generate & Analyze"
3. Auto-calculated metrics update in real-time
4. Submit to analyze accounts immediately

## Generated Account Patterns

### Bot Pattern (🤖 Following >> Followers)
- Followers: 0-50 (minimal)
- Following: 1000-10000 (aggressive)
- Follow Ratio: High (333x, 1000x+)
- Posts: 5000-50000
- No profile image (60% of bots)
- Not verified

### Real User Pattern (👤 Followers ≈ Following or Followers >> Following)
- Followers: 50-5000
- Following: 50-1000
- Follow Ratio: 0.5-10x (balanced)
- Posts: 100-5000
- Has profile image (90% of real)
- 5% verified

### Celebrity Pattern (👑 Followers >> Following)
- Followers: 100K+
- Following: 1K-10K (selective)
- Follow Ratio: Very low (0.01-0.1x)
- Posts: 1000-5000
- Complete profile
- Often verified

## Integration Points

- Seamlessly integrates with existing dashboard
- Results appear in Results Table immediately
- Security events logged for all generations
- Batch analysis ready after generation

## Next Steps

1. Open dashboard at **http://localhost:5173**
2. Click **Add Account** button
3. Try **Gen 50** for quick sample
4. Or use **Manual Entry** to test specific account patterns
5. Click **Secure Analyze** to process

---

*Beautiful, minimal design with powerful pattern generation for testing bot detection!*
