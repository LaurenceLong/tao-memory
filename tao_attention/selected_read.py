"""Selected fine N/D with reconstructed BF16 channels and bounded adjoint scratch."""
import torch
import triton
import triton.language as tl
from .indirect_fine import _selected_bounds, _sum_selected


@triton.jit
def _numerator(HV, W, PAGE, C, ORDER, START, END, PAIRS,
               Q:tl.constexpr,H:tl.constexpr,M:tl.constexpr,S:tl.constexpr,
               P:tl.constexpr,D:tl.constexpr,BANK_B:tl.constexpr,
               C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,C4:tl.constexpr,
               BLOCK:tl.constexpr):
    group=tl.program_id(0)
    e=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    d=tl.arange(0,D)
    offsets=tl.arange(0,BLOCK)
    segment=group%S; head=(group//S)%H; batch=group//(S*H)
    first,last=tl.load(START+group),tl.load(END+group)
    if first<last:
        c=tl.load(C+(batch%BANK_B)*C0+head*C1+segment*C2+d[:,None]*C3+e[None,:]*C4)
        w=tl.load(W+((batch*H+head)*S+segment)*D+d)
        for k in range(first,last,BLOCK):
            pos=k+offsets
            pair=tl.load(ORDER+(batch*H+head)*Q*M+pos,pos<last,other=0)
            query,slot=pair//M,pair%M
            row=((batch*Q+query)*H+head)*M+slot
            page=tl.load(PAGE+row,pos<last,other=0)
            hv=tl.load(HV+(((batch*Q+query[:,None])*H+head)*P+page[:,None])*D+d[None,:],
                       pos[:,None]<last,other=0)
            a=(hv.to(tl.float32)*w[None,:].to(tl.float32)).to(tl.bfloat16)
            contribution=tl.dot(a,c)
            tl.store(PAIRS+row[:,None]*D+e[None,:],contribution,pos[:,None]<last)


@triton.jit
def _denominator(HV,W,IDS,PAGE,DEN,Q:tl.constexpr,H:tl.constexpr,M:tl.constexpr,
                 S:tl.constexpr,P:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0)
    head=row%H; query=(row//H)%Q; batch=row//(Q*H)
    d=tl.arange(0,D)
    total=tl.full((),0,tl.float32)
    for slot in range(M):
        selected=row*M+slot
        fine,page=tl.load(IDS+selected),tl.load(PAGE+selected)
        hv=tl.load(HV+(row*P+page)*D+d)
        w=tl.load(W+((batch*H+head)*S+fine)*D+d)
        a=(hv.to(tl.float32)*w.to(tl.float32)).to(tl.bfloat16)
        # Preserve channel reduction -> BF16 -> slot reduction -> BF16.
        channel_sum=tl.sum(a.to(tl.float32),0).to(tl.bfloat16)
        total+=channel_sum.to(tl.float32)
    tl.store(DEN+row,total)


@triton.jit
def _channel_adjoint(C,GN,GD,ORDER,START,END,DA,
                      Q:tl.constexpr,H:tl.constexpr,M:tl.constexpr,S:tl.constexpr,
                      D:tl.constexpr,BANK_B:tl.constexpr,
                      C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,C4:tl.constexpr,
                      BLOCK:tl.constexpr):
    group=tl.program_id(0)
    d=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    e=tl.arange(0,D); offsets=tl.arange(0,BLOCK)
    segment=group%S; head=(group//S)%H; batch=group//(S*H)
    first,last=tl.load(START+group),tl.load(END+group)
    if first<last:
        c=tl.load(C+(batch%BANK_B)*C0+head*C1+segment*C2+d[None,:]*C3+e[:,None]*C4)
        for k in range(first,last,BLOCK):
            pos=k+offsets
            pair=tl.load(ORDER+(batch*H+head)*Q*M+pos,pos<last,other=0)
            query,slot=pair//M,pair%M
            g=tl.load(GN+((batch*Q+query[:,None])*H+head)*D+e[None,:],pos[:,None]<last,other=0)
            numerator_da=tl.dot(g,c).to(tl.bfloat16)
            denominator_da=tl.load(GD+(batch*Q+query)*H+head,pos<last,other=0)
            da=(numerator_da.to(tl.float32)+denominator_da[:,None].to(tl.float32)).to(tl.bfloat16)
            row=((batch*Q+query)*H+head)*M+slot
            tl.store(DA+row[:,None]*D+d[None,:],da,pos[:,None]<last)


@triton.jit
def _hv_adjoint(DA,W,IDS,PAGE,DHV,Q:tl.constexpr,H:tl.constexpr,M:tl.constexpr,
                 S:tl.constexpr,P:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0)
    page=row%P; qh=row//P; head=qh%H; batch=qh//(Q*H)
    d=tl.arange(0,D)
    acc=tl.full((D,),0,tl.bfloat16)
    for slot in range(M):
        selected=qh*M+slot
        selected_page=tl.load(PAGE+selected)
        if selected_page==page:
            fine=tl.load(IDS+selected)
            da=tl.load(DA+selected*D+d)
            w=tl.load(W+((batch*H+head)*S+fine)*D+d)
            product=(da.to(tl.float32)*w.to(tl.float32)).to(tl.bfloat16)
            # BF16 scatter-add boundary; deterministic selected-slot order.
            acc=(acc.to(tl.float32)+product.to(tl.float32)).to(tl.bfloat16)
    tl.store(DHV+row*D+d,acc)


@triton.jit
def _weight_adjoint(DA,HV,PAGE,ORDER,START,END,DW,
                     Q:tl.constexpr,H:tl.constexpr,M:tl.constexpr,S:tl.constexpr,
                     P:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    group=tl.program_id(0)
    d=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    segment=group%S; head=(group//S)%H; batch=group//(H*S)
    offsets=tl.arange(0,BLOCK)
    first,last=tl.load(START+group),tl.load(END+group)
    total=tl.full((BLOCK,),0,tl.float32)
    # Within each query the original TopM yields unique fine IDs. Across
    # queries, including R16 repeats, each BF16 product contributes once.
    for k in range(first,last,BLOCK):
        pos=k+offsets
        pair=tl.load(ORDER+(batch*H+head)*Q*M+pos,pos<last,other=0)
        query,slot=pair//M,pair%M
        row=((batch*Q+query)*H+head)*M+slot
        page=tl.load(PAGE+row,pos<last,other=0)
        da=tl.load(DA+row[:,None]*D+d[None,:],pos[:,None]<last,other=0)
        hv=tl.load(HV+(((batch*Q+query[:,None])*H+head)*P+page[:,None])*D+d[None,:],pos[:,None]<last,other=0)
        product=(da.to(tl.float32)*hv.to(tl.float32)).to(tl.bfloat16)
        total+=tl.sum(product.to(tl.float32),axis=0)
    tl.store(DW+group*D+d,total)


@triton.jit
def _matrix_adjoint(HV,W,PAGE,GN,ORDER,START,END,DC,
                     Q:tl.constexpr,H:tl.constexpr,M:tl.constexpr,S:tl.constexpr,
                     P:tl.constexpr,D:tl.constexpr,B:tl.constexpr,BANK_B:tl.constexpr,
                     BLOCK:tl.constexpr):
    group=tl.program_id(0)
    d=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    e=tl.program_id(2)*BLOCK+tl.arange(0,BLOCK)
    offsets=tl.arange(0,BLOCK)
    segment=group%S; head=(group//S)%H; bank_batch=group//(S*H)
    total=tl.full((BLOCK,BLOCK),0,tl.float32)
    for batch in range(bank_batch,B,BANK_B):
        tile_group=(batch*H+head)*S+segment
        first,last=tl.load(START+tile_group),tl.load(END+tile_group)
        acc=tl.full((BLOCK,BLOCK),0,tl.float32)
        w=tl.load(W+((batch*H+head)*S+segment)*D+d)
        for k in range(first,last,BLOCK):
            pos=k+offsets
            pair=tl.load(ORDER+(batch*H+head)*Q*M+pos,pos<last,other=0)
            query,slot=pair//M,pair%M
            row=((batch*Q+query)*H+head)*M+slot
            page=tl.load(PAGE+row,pos<last,other=0)
            hv=tl.load(HV+(((batch*Q+query[None,:])*H+head)*P+page[None,:])*D+d[:,None],pos[None,:]<last,other=0)
            a=(hv.to(tl.float32)*w[:,None].to(tl.float32)).to(tl.bfloat16)
            g=tl.load(GN+((batch*Q+query[:,None])*H+head)*D+e[None,:],pos[:,None]<last,other=0)
            acc+=tl.dot(a,g)
        total+=acc.to(tl.bfloat16).to(tl.float32)
    output=(((bank_batch*H+head)*S+segment)*D+d[:,None])*D+e[None,:]
    tl.store(DC+output,total)


class _SelectedRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,hv,weight,ids,pages,fine_c):
        hv,weight,ids,pages=hv.contiguous(),weight.contiguous(),ids.contiguous(),pages.contiguous()
        b,q,h,m=ids.shape;d=hv.shape[-1];p=hv.shape[3];s=fine_c.shape[2];bank_b=fine_c.shape[0]
        by_head=ids.permute(0,2,1,3).reshape(b,h,q*m).contiguous()
        sorted_ids,order=torch.sort(by_head,dim=-1,stable=True)
        starts=torch.empty((b,h,s),dtype=torch.int64,device=ids.device)
        ends=torch.empty_like(starts)
        _selected_bounds[(triton.cdiv(b*h*s,128),)](sorted_ids.contiguous(),starts,ends,q*m,s,b*h*s,(q*m).bit_length(),128)
        pairs=torch.empty((b,q,h,m,d),dtype=torch.float32,device=hv.device)
        numerator=torch.empty((b,q,h,d),dtype=hv.dtype,device=hv.device)
        denominator=torch.empty((b,q,h,1),dtype=hv.dtype,device=hv.device)
        _numerator[(b*h*s,triton.cdiv(d,32))](hv,weight,pages,fine_c,order,starts,ends,pairs,q,h,m,s,p,d,bank_b,*fine_c.stride(),32)
        _sum_selected[(b*q*h,)](pairs,numerator,m,d)
        _denominator[(b*q*h,)](hv,weight,ids,pages,denominator,q,h,m,s,p,d)
        ctx.save_for_backward(hv,weight,ids,pages,fine_c,order,starts,ends)
        return numerator,denominator

    @staticmethod
    def backward(ctx,grad_numerator,grad_denominator):
        hv,weight,ids,pages,fine_c,order,starts,ends=ctx.saved_tensors
        gn,gd=grad_numerator.contiguous(),grad_denominator.contiguous()
        b,q,h,m=ids.shape;d=hv.shape[-1];p=hv.shape[3];s=fine_c.shape[2];bank_b=fine_c.shape[0]
        da=torch.empty((b,q,h,m,d),dtype=hv.dtype,device=hv.device)
        _channel_adjoint[(b*h*s,triton.cdiv(d,32))](fine_c,gn,gd,order,starts,ends,da,q,h,m,s,d,bank_b,*fine_c.stride(),32)
        dhv=torch.empty_like(hv);dw=torch.empty_like(weight)
        dc=torch.empty(fine_c.shape,dtype=fine_c.dtype,device=fine_c.device)
        _hv_adjoint[(b*q*h*p,)](da,weight,ids,pages,dhv,q,h,m,s,p,d)
        _weight_adjoint[(b*h*s,triton.cdiv(d,32))](da,hv,pages,order,starts,ends,dw,q,h,m,s,p,d,32)
        _matrix_adjoint[(bank_b*h*s,triton.cdiv(d,32),triton.cdiv(d,32))](hv,weight,pages,gn,order,starts,ends,dc,q,h,m,s,p,d,b,bank_b,32)
        return dhv,dw,None,None,dc


def selected_fine_read(hv,segment_weight,selected_id,selected_page,fine_c):
    return _SelectedRead.apply(hv,segment_weight,selected_id,selected_page,fine_c)
