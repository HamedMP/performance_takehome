# Problem History and Benchmarks

## Source
This problem is Anthropic's performance engineering take-home challenge, released publicly.

## Key Benchmarks (cycles)
| Cycles | Achievement |
|--------|-------------|
| 2164 | Claude Opus 4 after many hours in test-time compute harness |
| 1790 | Claude Opus 4.5 casual session (matches best human 2hr performance) |
| 1579 | Claude Opus 4.5 after 2 hours in test-time compute harness |
| 1548 | Claude Sonnet 4.5 after many hours of test-time compute |
| **1487** | **Claude Opus 4.5 after 11.5 hours in harness** |
| 1363 | Claude Opus 4.5 in improved test-time compute harness |
| ??? | Fastest human solution - "substantially exceeds" Claude's best |

## Key Insights from Blog Post

1. **Novel algorithmic approaches win**: Claude found an optimization the author hadn't thought of - "transposing the entire computation rather than figuring out how to transpose the data"

2. **First principles reasoning matters**: The author found solutions "from first principles" while Claude draws on "a larger toolbox of experience"

3. **Debugging tools are valuable**: "Building debugging tools is part of what's being tested" - can build interactive debugger or use well-crafted print statements

4. **Human advantage exists at long time horizons**: "Human experts retain an advantage over current models at sufficiently long time horizons"

## Implications for Optimization

The fact that 1487 cycles is achievable means there ARE optimizations beyond what we've found.

Possible directions:
1. **Algorithmic transformation**: Like "transposing computation" - maybe restructure the entire approach
2. **Better analysis tools**: Build visualization to understand where cycles go
3. **Challenge assumptions**: Question fundamental structure of current solution
4. **Look for patterns**: What makes this problem "out of distribution"?

## Current State
- Current: 2,257 cycles
- Target: < 1,487 cycles
- Gap: 770 cycles (33% reduction needed)

The gap is substantial but clearly achievable given the benchmarks.
