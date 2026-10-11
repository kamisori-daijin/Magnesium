import asyncio
from pathlib import Path
from coreai.runtime import AIModel, SpecializationOptions, ComputeUnitKind
from coreai_torch.debugging.compute_plan import ComputePlan
from coreai_torch.debugging.debug_info import parse_debug_infos


async def analyze_compute_plan():
    """
    Load an AIModel and analyze its compute plan with ANE as preferred compute unit.
    
    Args:
        model_path: Path to the .aimodel file
    """
    # Load the AIModel with ANE as preferred compute unit
    model_path = Path("ane_raytracer.aimodel")
    
    # ANE を優先的に使うように設定
    specialization_options = SpecializationOptions.from_preferred_compute_unit_kind(
        ComputeUnitKind.neural_engine()
    )
    # Debug 情報を有効化
    specialization_options = specialization_options.with_debug(enabled=True)
    
    model = await AIModel.load(model_path, specialization_options)
    
    # Extract debug info records from the model
    debug_info_records = parse_debug_infos(model._debug_infos)
    
    # Create the ComputePlan
    compute_plan = ComputePlan(debug_info_records=debug_info_records)
    
    # Export to dictionary
    data = compute_plan.to_dict()
    print(f"Total entries: {data['entry_count']}")
    print(f"Device histogram: {data['device_histogram']}")
    
    # 各エントリの詳細を見る
    for entry in data['entries'][:10]:  # 最初の10個
        print(f"Operation {entry['operation_id']}: {entry['devices']} ({entry['residency']})")


if __name__ == "__main__":
    asyncio.run(analyze_compute_plan())