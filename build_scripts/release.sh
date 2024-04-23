# Usage: ./build_scripts/release.sh

RELEASE_VERSION=$(bump-my-version show current_version)
MAIN=main
BUMP_BRANCH=feature/bump-version-$RELEASE_VERSION

echo "Checking out $MAIN branch"
git checkout $MAIN
git pull origin $MAIN

echo "Creating tag $RELEASE_BRANCH"
git tag -a "$RELEASE_VERSION" -m "Release ${RELEASE_VERSION}"
git push --tags origin $MAIN

echo "Bumping to next patch version"
git checkout -b $BUMP_BRANCH
bump-my-version bump --commit patch
git push origin $BUMP_BRANCH

echo "Now, please create a PR for $BUMP_BRANCH and the release for tag $RELEASE_VERSION."
