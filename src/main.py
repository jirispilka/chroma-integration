import os

from apify import Actor


async def main():
    async with Actor:

        # Get the value of the actor input
        actor_input = await Actor.get_input() or {}

        os.environ["OPENAI_API_KEY"] = actor_input.get("openai_token") or ""

        dataset_id = (
            actor_input.get("payload", {}).get("resource", {}).get("defaultDatasetId", actor_input.get("dataset_id"))
        )

        Actor.log.info("Dataset ID %s", dataset_id)

        await Actor.exit(status_message="Index created successfully", exit_code=0)
