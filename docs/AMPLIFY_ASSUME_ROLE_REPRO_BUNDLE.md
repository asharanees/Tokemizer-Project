# Amplify AssumeRole Failure Repro Bundle

Date: 2026-03-03  
Account: `980874804229`  
Region: `us-east-1`

## App and Branches

- App ID: `d2qpkpwobax8e6`
- App ARN: `arn:aws:amplify:us-east-1:980874804229:apps/d2qpkpwobax8e6`
- App name: `Tokemizer-Project-Frontend`
- Repository: `https://github.com/nowusman/tokemizer`
- Default domain: `d2qpkpwobax8e6.amplifyapp.com`
- Current IAM service role ARN: `arn:aws:iam::980874804229:role/service-role/Tokemizer-Amplify-Role-20260303121004`

Branch state:
- `main` (PRODUCTION): failed deploys
- `dev` (DEVELOPMENT, display name `staging`): failed deploys

## Fresh Role Test Performed

Created a brand-new role and switched the app to it:

- Role ARN: `arn:aws:iam::980874804229:role/service-role/Tokemizer-Amplify-Role-20260303121004`
- Path: `/service-role/`
- Trust policy principal: `amplify.amazonaws.com`
- Trust actions: `sts:AssumeRole`, `sts:TagSession`
- Attached managed policies:
  - `arn:aws:iam::aws:policy/AdministratorAccess-Amplify`
  - `arn:aws:iam::aws:policy/service-role/AmplifyBackendDeployFullAccess`

## Repro Result (Still Fails)

Main branch job:
- `branch=main`, `jobId=7`, `status=FAILED`

Dev branch job:
- `branch=dev`, `jobId=2`, `status=FAILED`

Build log excerpt:

```
[ERROR]: !!! Unable to assume specified IAM Role. Please ensure the selected IAM Role has sufficient permissions and the Trust Relationship is configured correctly.
```

## Additional Context Checks

- Amplify app region is `us-east-1`.
- Role and app are in same AWS account.
- Organizations is enabled for the account.
- No SCPs were found directly attached to this account or root in current query.
- CloudTrail trails are not configured (`describe-trails` returned empty), limiting deny-event visibility.

## AWS Support Request (Suggested)

Please investigate why AWS Amplify cannot assume IAM service roles in account `980874804229` for app `d2qpkpwobax8e6` in `us-east-1`.

Include these identifiers:
- App: `d2qpkpwobax8e6`
- Role: `arn:aws:iam::980874804229:role/service-role/Tokemizer-Amplify-Role-20260303121004`
- Failed jobs: `main/7`, `dev/2`
- Error string: `Unable to assume specified IAM Role`

Ask support to verify:
- internal service-side role assumption path for Amplify in this account,
- hidden account-level restrictions/guardrails,
- any regional service control anomalies affecting Amplify STS assume role.
