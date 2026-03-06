# torii-image

A [Torii](https://github.com/haratosan/torii) extension that generates or edits images via AI using [OpenRouter](https://openrouter.ai/).

## Requirements

- Go 1.24+
- An [OpenRouter](https://openrouter.ai/) API key

## Installation

Clone this repo into your Torii extensions directory and build:

```sh
cd torii/extensions
git clone https://github.com/haratosan/torii-image.git
cd torii-image && go build .
```

Torii will automatically detect the extension on the next start.

## Configuration

| Variable                   | Description                              | Default                                    |
|----------------------------|------------------------------------------|--------------------------------------------|
| `TORII_OPENROUTER_API_KEY` | OpenRouter API key                       | *(required)*                               |
| `TORII_IMAGE_MODEL`        | Model to use for image generation/editing | `google/gemini-3.1-flash-image-preview`    |

Set these in your `config.yaml` under `extensions.env` or as environment variables.

## Usage

- **Generate**: Send a text prompt like "Draw a cat in a spaceship" and the extension generates an image.
- **Edit**: Send a photo to the bot with a text prompt like "Make the background blue" and the extension edits the image accordingly.

## License

MIT
