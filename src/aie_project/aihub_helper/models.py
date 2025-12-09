from typing import List, Literal, Union, Iterator

from pydantic import BaseModel, Field


class AIHubFile(BaseModel):
    type: Literal["file"] = "file"
    name: str
    size: str
    file_sn: str = Field(..., description="The unique File Serial Number used for downloading")


class AIHubFolder(BaseModel):
    type: Literal["folder"] = "folder"
    name: str
    children: List[Union["AIHubFolder", "AIHubFile"]] = Field(default_factory=list)


    # method overrides for easier access
    def __len__(self) -> int:
        return len(self.children)

    def __iter__(self) -> Iterator[Union["AIHubFolder", "AIHubFile"]]:
        return iter(self.children)

    def __getitem__(self, index: int) -> Union["AIHubFolder", "AIHubFile"]:
        return self.children[index]


class AIHubDataset(BaseModel):
    dataset_name: str
    root_folder: AIHubFolder


    def __len__(self) -> int:
        return len(self.root_folder)

    def __iter__(self) -> Iterator[Union["AIHubFolder", "AIHubFile"]]:
        return iter(self.root_folder)

    def __getitem__(self, index: int) -> Union["AIHubFolder", "AIHubFile"]:
        return self.root_folder[index]


# Rebuild model to support recursive reference (Folder -> children -> Folder)
AIHubFolder.model_rebuild()
