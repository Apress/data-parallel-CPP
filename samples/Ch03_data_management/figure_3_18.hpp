// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

class handler {
  ...
      // Specifies event(s) that must be complete before the
      // action Copy to/from an accessor. Valid
      // combinations: Src: accessor,   Dest: shared_ptr
      // Src: accessor,   Dest: pointer
      // Src: shared_ptr  Dest: accessor
      // Src: pointer     Dest: accessor
      // Src: accesssor   Dest: accessor
      template <typename T_Src, typename T_Dst, int Dims,
                access::mode AccessMode,
                access::target AccessTarget,
                access::placeholder IsPlaceholder =
                    access::placeholder::false_t>
      void copy(accessor<T_Src, Dims, AccessMode,
                         AccessTarget, IsPlaceholder>
                    Src,
                shared_ptr_class<T_Dst> Dst);
  void copy(shared_ptr_class<T_Src> Src,
            accessor<T_Dst, Dims, AccessMode, AccessTarget,
                     IsPlaceholder>
                Dst);
  void copy(accessor<T_Src, Dims, AccessMode, AccessTarget,
                     IsPlaceholder>
                Src,
            T_Dst *Dst);
  void copy(const T_Src *Src,
            accessor<T_Dst, Dims, AccessMode, AccessTarget,
                     IsPlaceholder>
                Dst);
  template <typename T_Src, int Dims_Src,
            access::mode AccessMode_Src,
            access::target AccessTarget_Src, typename T_Dst,
            int Dims_Dst, access::mode AccessMode_Dst,
            access::target AccessTarget_Dst,
            access::placeholder IsPlaceholder_Src =
                access::placeholder::false_t,
            access::placeholder IsPlaceholder_Dst =
                access::placeholder::false_t>
  void copy(accessor<T_Src, Dims_Src, AccessMode_Src,
                     AccessTarget_Src, IsPlaceholder_Src>
                Src,
            accessor<T_Dst, Dims_Dst, AccessMode_Dst,
                     AccessTarget_Dst, IsPlaceholder_Dst>
                Dst);

  // Provides a guarantee that the memory object accessed by
  // the accessor is updated on the host after this action
  // executes.
  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder =
                access::placeholder::false_t>
  void update_host(accessor<T, Dims, AccessMode,
                            AccessTarget, IsPlaceholder>
                       Acc);
  ...
};
