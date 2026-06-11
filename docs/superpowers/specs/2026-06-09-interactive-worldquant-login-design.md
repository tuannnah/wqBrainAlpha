# Interactive WorldQuant Login Design

## Goal

Replace `brain_credentials.txt` authentication with an interactive console
login. Users enter their email and password every time the program starts. If
WorldQuant requires an additional QR verification step, the program opens the
verification page in the default browser and waits for the user to finish.

## Scope

- Keep the existing console application and PyInstaller executable.
- Prompt for the email with `input()`.
- Prompt for the password with `getpass.getpass()` so it is not echoed.
- Never save the email or password to disk.
- Remove `brain_credentials.txt` from runtime and packaging workflows.
- Support both immediate authentication and browser-based additional
  verification.

The feature does not automate QR scanning or browser interaction.

## Components

### Console input

`main.py` collects the email and password before constructing the WorldQuant
client. Empty values are rejected without making a network request.

Input collection is kept separate from authentication so it can be tested
without connecting to WorldQuant.

### Authentication client

`BrainBatchAlpha` receives the email and password as constructor arguments. It
creates a `requests.Session`, assigns HTTP Basic authentication, and posts to
`/authentication`.

The authentication method classifies the response as:

- authenticated;
- additional verification required;
- invalid credentials or another authentication failure.

No credential data is included in exceptions, logs, or console output.

### Additional verification

When the authentication response indicates additional verification, the client
extracts the verification URL from known response headers or JSON fields. URL
extraction is isolated in a helper so supported response shapes can be extended
without changing the console flow.

The program opens the URL with `webbrowser.open()`, also prints the URL as a
fallback, and asks the user to:

1. Open the verification page.
2. Scan the QR code and complete verification.
3. Return to the console and press Enter.

After Enter is pressed, the client repeats the authentication request using the
same session. A failed verification can be retried a limited number of times.
If no verification URL is present, the program reports that additional
verification is required but cannot be opened automatically.

This manual confirmation flow is preferred over indefinite API polling because
WorldQuant does not publish a stable public schema for the QR challenge
response.

## Data Flow

1. The user selects a run mode.
2. The application asks for email and password.
3. `BrainBatchAlpha` sends the authentication request.
4. Immediate success continues to dataset or submit operations.
5. A verification challenge opens the browser and pauses the console.
6. The user completes QR verification and presses Enter.
7. The application checks authentication again.
8. Success continues; exhausted retries terminate with a clear error.

Credentials remain only in process memory for the lifetime of the client.

## Error Handling

- Empty email or password: reject locally and ask again.
- Invalid credentials: show a generic authentication error.
- Network error or timeout: show a connection error and terminate cleanly.
- Browser cannot open: print the verification URL for manual use.
- Verification URL missing: report the unsupported response without exposing
  the response body if it might contain sensitive data.
- Verification incomplete: allow a small, fixed number of checks before
  failing.

All authentication requests use an explicit timeout.

## Packaging And Documentation

- Stop copying or creating `brain_credentials.txt` in `build.py` and
  `build_windows.py`.
- Remove it from the zipapp file list.
- Keep the ignore rule temporarily so existing local credential files cannot
  be committed accidentally.
- Update the README to describe interactive login and QR verification.
- Existing local or `dist` credential files are not deleted automatically.

## Testing

Tests use fake sessions and responses; they do not call the live WorldQuant
API.

Coverage includes:

- email and password are passed directly to HTTP Basic authentication;
- successful authentication;
- invalid credentials;
- extraction of verification URLs from supported response shapes;
- browser opening and confirmation before rechecking authentication;
- browser-open failure fallback;
- missing verification URL;
- verification retry limit;
- empty console credentials;
- build scripts no longer reference or create `brain_credentials.txt`.

Implementation follows test-driven development: each behavior is represented by
a failing test before production code is changed.
