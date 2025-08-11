from gooey import Gooey, GooeyParser


def create_checkbox_interface(selector_groups):
    """
    Create a checkbox interface based on the provided configuration.

    Args:
        selector_groups: List of dictionaries, each containing:
            - 'name': Display name for the group
            - 'items': List of items to select from
            - 'mode': 'single' or 'multiple'

    Returns:
        Dictionary with selected items for each group
    """

    @Gooey(program_name="Checkbox Selection Interface",
           default_size=(800, 700),
           navigation='TABBED')
    def main():
        # Create the parser
        parser = GooeyParser(description="Select items from the lists")

        # Create subparsers
        subparsers = parser.add_subparsers(dest='command')

        # Selection tab
        selection_parser = subparsers.add_parser('Selection')

        # Add each selector group
        for group_config in selector_groups:
            group_name = group_config['name']
            items = group_config['items']
            mode = group_config['mode']

            # Create the group
            if mode == 'single':
                group_label = f"{group_name} (select only one)"
            else:
                group_label = f"{group_name} (select any number)"

            arg_group = selection_parser.add_argument_group(group_label)

            # For single selection, create a mutually exclusive group
            if mode == 'single':
                mutex_group = arg_group.add_mutually_exclusive_group()
                target_group = mutex_group
            else:
                target_group = arg_group

            # Add items to the group
            for item in items:
                safe_item = str(item).lower().replace(" ", "_").replace("-", "_")
                arg_name = f"--{group_name.lower().replace(' ', '_')}_{safe_item}"
                target_group.add_argument(arg_name,
                                          metavar=str(item),
                                          action='store_true')

        # Parse the arguments
        args = parser.parse_args()

        # Process the selections
        if args.command == 'Selection':
            # Gather all selections
            selections = {}

            for group_config in selector_groups:
                group_name = group_config['name']
                items = group_config['items']
                mode = group_config['mode']

                selected_items = []
                for item in items:
                    safe_item = str(item).lower().replace(" ", "_").replace("-", "_")
                    arg_name = f"{group_name.lower().replace(' ', '_')}_{safe_item}"
                    if getattr(args, arg_name, False):
                        selected_items.append(item)

                selections[group_name] = selected_items

            return selections

        return {}

    # Run the Gooey interface
    return main()

