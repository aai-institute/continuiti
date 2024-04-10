# Usage: ./build_scripts/release.sh

RELEASE_VERSION=$(bump-my-version show current_version)
MAIN=main

echo "Checking out $MAIN branch"
git checkout $MAIN
git pull origin $MAIN

echo "Creating tag $RELEASE_BRANCH"
git tag -a "$RELEASE_VERSION" -m "Release ${RELEASE_VERSION}"
git push --tags origin $MAIN

echo "Bumping to next patch version"
bump-my-version bump --commit patch

git push origin $MAIN
