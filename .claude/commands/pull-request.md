# Write Pull Request Description

Target branch: $ARGUMENTS:target-branch
Additional context: $ARGUMENTS:context

Write a pull request (PR) description markdown file for merging the current branch into a target branch.

## Arguments

- **target-branch** (optional): The branch to merge into. Defaults to `main` if not provided.
- **context** (optional): Rationale or additional context to incorporate into the PR description.

Example usage:

- `/pull-request` - merge into `main`, no additional context
- `/pull-request --target-branch develop` - merge into `develop`
- `/pull-request --context "Added caching for performance"` - merge into `main` with context
- `/pull-request --target-branch develop --context "Refactored auth flow"` - merge into `develop` with context

## Instructions

1. **Determine Target Branch**
   - Use the target branch from "Target branch:" above
   - If empty or not provided, use `main` as the target branch
   - Verify the target branch exists by running `git rev-parse --verify <target-branch>`

2. **Gather Branch Information**
   - Run `git branch --show-current` to get the current branch name
   - Run `git log <target-branch>..HEAD --oneline` to list all commits on this branch
   - Run `git diff <target-branch>...HEAD --stat` to get an overview of changed files
   - Run `git diff <target-branch>...HEAD` to understand the detailed changes

3. **Analyze the Changes**
   - Identify the main purpose/goal of this branch
   - Group related changes into logical categories
   - Note any breaking changes or migration requirements
   - Identify key implementation decisions

4. **Write the PR Description**
   - Create a markdown file at `PR_DESCRIPTION.md` in the repository root
   - Mention the target branch in the description if it's not `main`
   - If additional context is provided above, incorporate it to explain the motivation

5. **Description Format**

```markdown
# [PR Title: concise, descriptive title following conventional commit style]

## Summary

[1-3 sentences describing the overall purpose and impact of this PR]

## Changes

[Bulleted list of changes, grouped by category if applicable]

- **Category 1**
  - Change description
  - Change description

- **Category 2**
  - Change description

## Testing

[How the changes were tested or should be tested]

## Notes

[Any additional context, caveats, or follow-up items - omit if not applicable]
```

## Guidelines

- **Concise but comprehensive**: Cover all significant changes without excessive detail
- **Readable structure**: Use clear headings and bullet points
- **Focus on "what" and "why"**: Explain both the changes and their motivation
- **Highlight impact**: Call out breaking changes, new dependencies, or configuration changes
- **Skip trivial details**: Don't list every file or minor refactoring unless relevant
